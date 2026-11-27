"""Recipe B on verl: verl's own fully-async GRPO trainer, fed and published through the pool.

verl 0.8 ships an asynchronous trainer (verl.experimental.fully_async_policy): a FullyAsyncTrainer
takes rollouts from a MessageQueue that a FullyAsyncRollouter fills, and a checkpoint engine pushes
new weights to the rollouter's vLLM servers. All three are Ray actors of one Ray cluster. Here the
trainer is verl's, unchanged in what it computes (FSDP actor with LoRA, GRPO advantages, the PPO-clip
loss in bypass mode: old log-probs are the rollout's own, so the ratio corrects for stale rollouts),
and the other two are pool primitives, so rollers can run on any machine of the pool:

    verl piece                       pool primitive
    MessageQueue (Ray actor)         a pool queue of rollout tasks with max_lag against the policy ref
    checkpoint engine (NCCL/IPC)     a pool ref: each step's LoRA adapter is the ref's next version
    FullyAsyncRollouter (vLLM)       a pool actor of vLLM rollers subscribed to the ref (roller.py)

PoolTrainer overrides only where the samples come from (_get_samples_from_queue), where the weights
go (_fit_update_weights), and turns off the rollouter-side validation and checkpointing. Per step it
appends the same metrics as Demo B's trainer (reward, lag, wait and train seconds) to
/outputs/metrics.jsonl, plus some of verl's own (pg loss, clip fraction, KL to the rollout policy).

Run: python pool_verl.py <hydra overrides>   (run.sh has the settings)
"""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import verl  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer, TrainingStopException  # noqa: E402
from verl.experimental.separation.engine_workers import DetachActorWorker  # noqa: E402
from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402
from verl.utils.debug import marked_timer  # noqa: E402

import common  # noqa: E402  (the task, the reward and the prompt template, shared with the rollers)

OUT = os.environ.get("POOL_OUTPUTS", "/outputs")
QUEUE, POLICY = os.environ.get("QUEUE", "rollouts"), os.environ.get("POLICY", "grpo/policy")
STEPS = int(os.environ.get("STEPS", "30"))
TASKS_PER_STEP = int(os.environ.get("TASKS_PER_STEP", "8"))
PROMPTS_PER_TASK = int(os.environ.get("PROMPTS_PER_TASK", "2"))
AHEAD = int(os.environ.get("AHEAD", "1"))  # steps of tasks kept queued beyond the current one (0: lock-step)
MAX_LAG = int(os.environ.get("MAX_LAG", "1"))  # policy versions a rollout may be behind when it is used
VERL_METRICS = ("actor/pg_loss", "actor/pg_clipfrac", "actor/ppo_kl", "actor/grad_norm", "response_length/mean",
                "timing_s/update_actor", "timing_s/param_sync")


class PoolActorWorker(DetachActorWorker):
    """verl's actor worker, plus a way to write its LoRA adapter as a PEFT folder (what vLLM loads)."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_lora(self, folder: str) -> int:
        from safetensors.torch import save_file

        # The same call verl's own weight sync makes; with LoRA it yields only the adapter's tensors,
        # under PEFT's names (get_peft_model_state_dict).
        params, _ = self.actor.engine.get_per_tensor_param(layered_summon=False, base_sync_done=True)
        tensors = {name: tensor.detach().to(torch.bfloat16).contiguous().cpu() for name, tensor in params}
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return len(tensors)
        module = self.actor.engine.module
        peft_model = getattr(module, "_fsdp_wrapped_module", module)
        os.makedirs(folder, exist_ok=True)
        peft_model.peft_config["default"].save_pretrained(folder)
        save_file(tensors, os.path.join(folder, "adapter_model.safetensors"))
        return len(tensors)


class PoolTrainer(FullyAsyncTrainer.__ray_actor_class__):
    """verl's FullyAsyncTrainer with the pool in place of its MessageQueue, rollouter and weight sync."""

    def setup_pool(self) -> None:
        self.policy_version = 0  # the pool ref's version: what the rollers tag their results with
        self.next_seed, self.after, self.pending, self.stale = 0, 0, [], 0
        self.wait_s, self.batch_info = 0.0, {}
        self.metrics_file = open(os.path.join(OUT, "metrics.jsonl"), "w")

    # ---- samples: from the pool queue instead of verl's MessageQueue ------------------------------

    def _top_up(self) -> None:
        counts = pool_job.stats(QUEUE)["counts"]
        want = TASKS_PER_STEP * (1 + AHEAD) - counts["queued"] - counts["running"] - len(self.pending)
        if want > 0:
            pool_job.add_tasks(QUEUE, [{"inputs": {"seed": self.next_seed + i, "prompts": PROMPTS_PER_TASK},
                                        "key": f"s{self.next_seed + i}"} for i in range(want)])
            self.next_seed += want

    async def _get_samples_from_queue(self):
        if self.global_steps > STEPS:
            return None, None  # verl's stop signal (it raises TrainingStopException)
        self._top_up()
        started = time.time()
        while True:
            # The queue checks max_lag when a result arrives; check it again when it is used, since the
            # policy may have moved on while the result waited here (Demo B: this matters for reward).
            fresh = [r for r in self.pending if self.policy_version - r["computed_with"] <= MAX_LAG]
            self.stale += len(self.pending) - len(fresh)
            self.pending = fresh
            if len(self.pending) >= TASKS_PER_STEP:
                break
            reply = pool_job.results(QUEUE, after=self.after, wait=20)
            self.after = reply["next"]
            self.pending += reply["results"]
        self.wait_s = time.time() - started
        used, self.pending = self.pending[:TASKS_PER_STEP], self.pending[TASKS_PER_STEP:]
        return 0, self._to_dataproto(used)

    def _to_dataproto(self, results: list[dict]) -> DataProto:
        """Rollout groups as the batch verl's agent loop would have produced: left-padded prompts,
        right-padded responses, rollout log-probs, and the reward on each response's last token.
        Rewards are recomputed here from the completions rather than taken from the roller."""
        tokenizer, rows = self.tokenizer, []
        for result in results:
            seed = int(result["key"][1:])
            for index, group in enumerate(result["result"]["groups"]):
                question, answer = common.problem(seed, index)
                prompt = tokenizer(common.render(tokenizer, question), add_special_tokens=False)["input_ids"]
                for sample in group["samples"]:
                    ids = sample["ids"] or [tokenizer.eos_token_id]
                    logprobs = (sample["logprobs"] + [0.0] * len(ids))[: len(ids)]
                    score = common.reward(tokenizer.decode(ids, skip_special_tokens=True), answer)
                    rows.append((prompt, ids, logprobs, score, f"{result['key']}-{index}", result["computed_with"]))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width_p, width_r, n = max(len(r[0]) for r in rows), max(len(r[1]) for r in rows), len(rows)
        prompts = torch.full((n, width_p), pad, dtype=torch.long)
        responses = torch.full((n, width_r), pad, dtype=torch.long)
        prompt_mask = torch.zeros((n, width_p), dtype=torch.long)
        response_mask = torch.zeros((n, width_r), dtype=torch.long)
        rollout_log_probs = torch.zeros((n, width_r), dtype=torch.float32)
        rm_scores = torch.zeros((n, width_r), dtype=torch.float32)
        for i, (prompt, ids, logprobs, score, _, _) in enumerate(rows):
            prompts[i, width_p - len(prompt):] = torch.tensor(prompt)
            prompt_mask[i, width_p - len(prompt):] = 1
            responses[i, : len(ids)] = torch.tensor(ids)
            response_mask[i, : len(ids)] = 1
            rollout_log_probs[i, : len(ids)] = torch.tensor(logprobs)
            rm_scores[i, len(ids) - 1] = score
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=1) - 1, min=0)
        versions = np.array([r[5] - 1 for r in rows], dtype=object)  # verl counts from 0, the ref from 1
        batch = DataProto.from_dict(
            tensors={"prompts": prompts, "responses": responses, "response_mask": response_mask,
                     "input_ids": torch.cat([prompts, responses], dim=1), "attention_mask": attention_mask,
                     "position_ids": position_ids, "rollout_log_probs": rollout_log_probs, "rm_scores": rm_scores},
            non_tensors={"uid": np.array([r[4] for r in rows], dtype=object),
                         "__num_turns__": np.full(n, 2, dtype=np.int32)},
            meta_info={"global_token_num": attention_mask.sum(-1).tolist(), "trajectory_param_versions": versions,
                       "reward_extra_keys": []},
        )
        lags = [self.policy_version - r["computed_with"] for r in results]
        self.batch_info = {"reward": float(np.mean([r[3] for r in rows])), "lag_max": max(lags),
                           "lag_mean": round(sum(lags) / len(lags), 2), "samples": n,
                           "machines": sorted({r["machine"] for r in results})}
        return batch

    # ---- weights: to the pool ref instead of verl's checkpoint engine ------------------------------

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return
        with marked_timer("timing_s/param_sync", self.timing_raw):
            folder = os.path.join(OUT, "policy-next")
            shutil.rmtree(folder, ignore_errors=True)
            self.actor_wg.export_lora(folder)
            self.policy_version = pool_job.publish_ref(POLICY, "policy-next", parent=self.policy_version)["version"]
            shutil.rmtree(folder, ignore_errors=True)  # published: do not keep every version in the outputs
        print(f"[PoolTrainer] published {POLICY} v{self.policy_version} "
              f"({self.timing_raw['timing_s/param_sync']:.2f}s)", flush=True)

    async def _fit_validate(self, val_before_train=False):
        return  # verl validates through its rollouter; here the rollers are pool jobs

    def _fit_save_checkpoint(self, force=False):
        return  # pool_job.save_checkpoint() would go here; short runs do not need it

    # ---- the loop ----------------------------------------------------------------------------------

    async def fit_step(self, batch_dict: dict = None):
        step, started = self.global_steps, time.time()
        await super().fit_step(batch_dict)
        record = {"step": step, "version": self.policy_version, "reward": round(self.batch_info["reward"], 4),
                  "lag_max": self.batch_info["lag_max"], "lag_mean": self.batch_info["lag_mean"],
                  "wait_s": round(self.wait_s, 2), "train_s": round(time.time() - started - self.wait_s, 2),
                  "at": round(time.time(), 1), "machines": self.batch_info["machines"], "stale_dropped": self.stale,
                  "samples": self.batch_info["samples"]}
        record.update({key: round(float(self.metrics[key]), 5) for key in VERL_METRICS if key in self.metrics})
        record.update({key: round(float(self.timing_raw[key]), 3) for key in ("timing_s/param_sync",) if key in self.timing_raw})
        print(json.dumps(record), flush=True)
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()

    async def fit(self):
        self.setup_pool()
        await self._fit_update_weights()  # v1: the untrained adapter, so the rollers start from the same LoRA
        self.global_steps = 1
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                break
        self.progress_bar.close()


@ray.remote(num_cpus=1)
class PoolTaskRunner:
    def run(self, config):
        from verl.experimental.separation.utils import create_resource_pool_manager
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.utils import Role
        from verl.utils import hf_tokenizer

        OmegaConf.resolve(config)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
        mapping = {Role.Actor: ray.remote(PoolActorWorker)}
        trainer = ray.remote(num_cpus=1)(PoolTrainer).remote(
            config=config, tokenizer=tokenizer, role_worker_mapping=mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Actor]),
            ray_worker_group_cls=RayWorkerGroup, device_name=config.trainer.device)
        ray.get(trainer.init_workers.remote())
        ray.get(trainer.set_total_train_steps.remote(STEPS))
        ray.get(trainer.fit.remote())


# verl's fully-async config folder is not a Python package, so hydra gets its path.
CONFIG = os.path.join(os.path.dirname(verl.__file__), "experimental", "fully_async_policy", "config")


@hydra.main(config_path=CONFIG, config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(migrate_legacy_reward_impl(config), task_runner_class=PoolTaskRunner)


if __name__ == "__main__":
    main()

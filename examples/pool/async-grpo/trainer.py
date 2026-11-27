"""The GRPO trainer: a LoRA policy on the base model, fed by rollers through a pool queue.

Each step it keeps the queue topped up (tasks for the steps ahead that the lag allows), takes the
first TASKS_PER_STEP accepted rollout groups, does one policy-gradient step with group-normalized
advantages and importance weights against the logprobs the rollouts were sampled with (clipped, as
in PPO; that corrects for a rollout made with a slightly older policy), and publishes the adapter
as the next version of the policy ref. The queue's max_lag (set by the driver) decides how old a
rollout may be: 0 makes it synchronous, which is the baseline.

Per step it records the mean reward, the policy lag of the rollouts used, and how long it waited
for rollouts (idle) versus trained, in /outputs/metrics.jsonl.
"""

import contextlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job  # noqa: E402
import pool_verify  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import common  # noqa: E402

QUEUE, POLICY = os.environ.get("QUEUE", "rollouts"), os.environ.get("POLICY", "grpo/policy")
STEPS = int(os.environ.get("STEPS", "30"))
TASKS_PER_STEP = int(os.environ.get("TASKS_PER_STEP", "8"))
PROMPTS_PER_TASK = int(os.environ.get("PROMPTS_PER_TASK", "2"))
AHEAD = int(os.environ.get("AHEAD", "1"))  # steps of tasks kept queued beyond the current one (0: lock-step)
MAX_LAG = int(os.environ.get("MAX_LAG", "1"))  # policy versions a rollout may be behind when it is used
LR = float(os.environ.get("LR", "5e-5"))
CLIP = 0.2
OUT = os.environ.get("POOL_OUTPUTS", "/outputs")

base = os.path.join(os.environ.get("POOL_INPUTS", "/inputs"), "base")
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def publish(parent: int) -> int:
    folder = os.path.join(OUT, "policy-next")
    shutil.rmtree(folder, ignore_errors=True)
    model.save_pretrained(folder)
    version = pool_job.publish_ref(POLICY, "policy-next", parent=parent)["version"]
    shutil.rmtree(folder, ignore_errors=True)  # published: do not keep every version in the outputs
    return version


def sequence_logprobs(prompt: str, ids: list[int]) -> torch.Tensor:
    """Log-probabilities of `ids` after `prompt`, under the current policy (one sample)."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokens = torch.tensor([prompt_ids + ids], device="cuda")
    logits = model(tokens).logits[0, len(prompt_ids) - 1:-1].float()
    return torch.log_softmax(logits, -1).gather(1, tokens[0, len(prompt_ids):, None])[:, 0]


CHECK_TOL = float(os.environ.get("CHECK_TOL", "0.5"))  # mean |logprob difference| a real rollout stays under
CHECK_SAMPLES = int(os.environ.get("CHECK_SAMPLES", "1"))  # samples per prompt whose logprobs (and proofs) are checked
TOPLOC = os.environ.get("TOPLOC") == "1"  # rollouts must carry TOPLOC proofs (roller.py with TOPLOC=1)

# The LoRA weights of the last few policy versions, on the CPU (a few MB each), so that a rollout made
# with an older version is checked against that version, not the current one. Without them a dishonest
# roller could dodge the checks by claiming its rollouts are one version old.
lora = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
snapshots: dict[int, list[torch.Tensor]] = {}


def remember(version: int) -> None:
    snapshots[version] = [p.detach().to("cpu", copy=True) for _, p in lora]
    for old in [v for v in snapshots if v < version - MAX_LAG]:
        del snapshots[old]


@contextlib.contextmanager
def policy_at(version: int):
    """The model as it was at `version` (one of the remembered ones) for the duration of the block."""
    current = [p.detach().clone() for _, p in lora]
    with torch.no_grad():
        for (_, p), saved in zip(lora, snapshots[version]):
            p.copy_(saved)
    try:
        yield
    finally:
        with torch.no_grad():
            for (_, p), saved in zip(lora, current):
                p.copy_(saved)


def verify(result: dict) -> list[str]:
    """Check a rollout before training on it (M6): rewards recomputed from the completions; for
    CHECK_SAMPLES samples per prompt, the logprobs recomputed under the policy version the rollout
    claims and, with TOPLOC=1, its TOPLOC proofs checked against the same forward pass's hidden states.
    Returns what is wrong (empty when it passes); the trainer reports the machine either way."""
    problems = []
    seed = int(str(result.get("key") or "s0")[1:])
    claimed = result["computed_with"]
    if claimed not in snapshots:  # stale ones are dropped before this; a version from the future is a lie
        return [f"logprobs: claims policy v{claimed}, which the trainer never published"]
    for index, group in enumerate(result["result"]["groups"]):
        question, answer = common.problem(seed, index)
        if group["question"] != question:
            return [f"reward: question {index} is not the one asked"]
        for sample in group["samples"]:
            text = tokenizer.decode(sample["ids"], skip_special_tokens=True)
            if abs(common.reward(text, answer) - sample["reward"]) > 1e-6:
                problems.append(f"reward: reported {sample['reward']}, the completion earns {common.reward(text, answer)}")
                break
        prompt_ids = tokenizer(common.render(tokenizer, question), add_special_tokens=False)["input_ids"]
        for sample in [s for s in group["samples"] if s["ids"]][:CHECK_SAMPLES]:
            with torch.no_grad(), policy_at(claimed):
                out = model(torch.tensor([prompt_ids + sample["ids"]], device="cuda"), output_hidden_states=True)
            logits = out.logits[0, len(prompt_ids) - 1:-1].float()
            ids = torch.tensor(sample["ids"], device="cuda")
            new = torch.log_softmax(logits, -1).gather(1, ids[:, None])[:, 0]
            gap = pool_verify.logprob_gap(sample["logprobs"], new)
            if gap > CHECK_TOL:
                problems.append(f"logprobs: mean difference {gap:.2f} from the policy's own")
            if TOPLOC:
                found = pool_verify.toploc_verify(out.hidden_states[-1][0], len(prompt_ids), sample.get("toploc") or [])
                if not found["ok"]:
                    problems.append(found["detail"])
    return problems


def train_step(results: list[dict]) -> float:
    model.train()
    optimizer.zero_grad()
    samples = [(group["question"], s) for r in results for group in r["result"]["groups"] for s in group["samples"]]
    count, rewards = len(samples), []
    for result in results:
        for group in result["result"]["groups"]:
            values = torch.tensor([s["reward"] for s in group["samples"]])
            rewards += values.tolist()
            advantages = (values - values.mean()) / (values.std() + 1e-4)
            prompt = common.render(tokenizer, group["question"])
            for sample, advantage in zip(group["samples"], advantages.tolist()):
                if not sample["ids"] or advantage == 0.0:
                    continue
                new = sequence_logprobs(prompt, sample["ids"])
                old = torch.tensor(sample["logprobs"][: len(new)], device="cuda")
                ratio = torch.exp(new - old)
                loss = -torch.min(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage).mean() / count
                loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return sum(rewards) / len(rewards)


version = publish(0)  # v1: the untrained adapter, so rollers start with the same LoRA path
remember(version)
seed, after, pending, stale, rejected = 0, 0, [], 0, 0
caught = {}  # failed checks by kind (a rollout can fail several)
metrics = open(os.path.join(OUT, "metrics.jsonl"), "w")
for step in range(1, STEPS + 1):
    stats = pool_job.stats(QUEUE)["counts"]
    waiting = stats["queued"] + stats["running"]
    want = TASKS_PER_STEP * (1 + AHEAD) - waiting - len(pending)
    if want > 0:
        pool_job.add_tasks(QUEUE, [{"inputs": {"seed": seed + i, "prompts": PROMPTS_PER_TASK}, "key": f"s{seed + i}"}
                                   for i in range(want)])
        seed += want
    started = time.time()
    while True:
        fresh = [r for r in pending if version - r["computed_with"] <= MAX_LAG]
        stale += len(pending) - len(fresh)  # accepted in time, but the policy moved on while they waited
        pending = fresh
        if len(pending) >= TASKS_PER_STEP:
            break
        reply = pool_job.results(QUEUE, after=after, wait=20)
        after = reply["next"]
        for result in reply["results"]:
            if version - result["computed_with"] > MAX_LAG:  # on time when it arrived, but the policy moved on
                stale += 1
                continue
            problems = verify(result)
            pool_job.report(result["machine"], not problems, problems[0].split(":")[0] if problems else "reward",
                            "; ".join(problems))
            if problems:
                rejected += 1
                for kind in sorted({p.split(":")[0] for p in problems}):
                    caught[kind] = caught.get(kind, 0) + 1
                print(f"rejected a rollout from {result['machine']}: {problems[0]}", flush=True)
            else:
                pending.append(result)
    waited = time.time() - started
    batch, pending = pending[:TASKS_PER_STEP], pending[TASKS_PER_STEP:]
    lag = [version - r["computed_with"] for r in batch]
    trained = time.time()
    mean_reward = train_step(batch)
    version = publish(version)
    remember(version)
    record = {"step": step, "version": version, "reward": round(mean_reward, 4), "lag_max": max(lag),
              "lag_mean": round(sum(lag) / len(lag), 2), "wait_s": round(waited, 2), "train_s": round(time.time() - trained, 2),
              "at": round(time.time(), 1), "machines": sorted({r["machine"] for r in batch}), "stale_dropped": stale, "rejected": rejected,
              "caught": caught}
    print(json.dumps(record), flush=True)
    metrics.write(json.dumps(record) + "\n")
    metrics.flush()

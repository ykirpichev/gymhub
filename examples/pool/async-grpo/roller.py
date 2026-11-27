"""An actor replica: generates GRPO rollouts with the newest policy version it has.

Loop: if the policy ref (POLICY) has a newer version, fetch it (the agent has usually pre-fetched
it: the actor subscribes to the ref) and switch LoRA adapters; lease a task (SEED, PROMPTS); sample
GROUP completions per prompt; score them; report the completions, their sampling logprobs and
rewards, tagged with the policy version they were computed with. The queue's max_lag decides
whether a late result still counts.

With TOPLOC=1 (the checked tier, M6) every completion also carries TOPLOC proofs of the policy's last
hidden states (pool_verify): one prefill of prompt + completion in the same vLLM engine, with a hook
on the final norm (so the engine runs without CUDA graphs). The trainer checks them.

FAKE makes this a dishonest machine, for the checked-tier demo: FAKE=1 claims every answer is right
and makes up logprobs; FAKE=fp8 serves the policy in fp8 to save memory, and FAKE=base serves the base
model without the policy's adapter. Those two are honest otherwise: their rewards are right, and
their logprobs and proofs are those of the model they really ran.
"""

import os
import sys
import time

sys.path.insert(0, os.environ["POOL_LIB"])
import pool_job  # noqa: E402

import common  # noqa: E402

QUEUE, POLICY = os.environ.get("QUEUE", "rollouts"), os.environ.get("POLICY", "grpo/policy")
GROUP, MAX_TOKENS = int(os.environ.get("GROUP", "8")), int(os.environ.get("MAX_TOKENS", "200"))
IDLE = float(os.environ.get("IDLE", "120"))
FAKE = os.environ.get("FAKE", "")  # "1": every reward 1.0, made-up logprobs; "fp8": the policy in fp8; "base": no LoRA
TOPLOC = os.environ.get("TOPLOC") == "1"
if TOPLOC:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # the engine in this process, for the hidden-state hook
    import pool_verify  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

base = os.path.join(os.environ.get("POOL_INPUTS", "/inputs"), "base")
# Replicas sharing a GPU must not profile memory at the same moment: vLLM would count the others'
# allocations as its own and find no room for the KV cache. Stagger the starts a little.
time.sleep(int(os.environ.get("POOL_JOB_ID", "j0")[-2:], 16) % 20)
llm = LLM(base, enable_lora=True, max_lora_rank=16, max_loras=2, max_model_len=1024,
          gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.08")), seed=int(time.time()) % 1000,
          enforce_eager=TOPLOC, quantization="fp8" if FAKE == "fp8" else None)
tokenizer = llm.get_tokenizer()
states = pool_verify.VllmStates(llm) if TOPLOC else None
sampling = SamplingParams(n=GROUP, temperature=1.0, max_tokens=MAX_TOKENS, logprobs=0)
version, adapter, last_work = 0, None, time.time()

while time.time() - last_work < IDLE:
    try:
        current = pool_job.ref(POLICY)["version"]
    except pool_job.PoolJobError:
        time.sleep(2)  # the trainer has not published the first policy yet
        continue
    if current != version:
        path, version = pool_job.fetch_ref(POLICY)
        adapter = None if FAKE == "base" else LoRARequest(f"policy-v{version}", version, path)  # "base": skips the policy
    tasks = pool_job.lease(QUEUE, n=1, wait=5)
    if not tasks:
        continue
    last_work = time.time()
    task = tasks[0]
    questions = [common.problem(task["inputs"]["seed"], i) for i in range(task["inputs"]["prompts"])]
    prompts = [tokenizer(common.render(tokenizer, q), add_special_tokens=False)["input_ids"] for q, _ in questions]
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sampling, lora_request=adapter, use_tqdm=False)
    groups = []
    for (question, answer), prompt_ids, output in zip(questions, prompts, outputs):
        samples = []
        for sample in output.outputs:
            logprobs = [round(next(iter(step.values())).logprob, 4) for step in sample.logprobs]
            reward = common.reward(sample.text, answer)
            if FAKE == "1":  # a dishonest machine, for the checked-tier test: claims every answer is right
                reward, logprobs = 1.0, [-0.01] * len(logprobs)
            entry = {"ids": list(sample.token_ids), "logprobs": logprobs, "reward": reward}
            if states is not None:
                entry["toploc"] = pool_verify.toploc_proofs(states(prompt_ids, entry["ids"], adapter), len(prompt_ids))
            samples.append(entry)
        groups.append({"question": question, "samples": samples})
    pool_job.result(QUEUE, task["id"], {"groups": groups}, computed_with=version)
print("no work for a while; stopping")

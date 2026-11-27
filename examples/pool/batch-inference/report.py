"""Throughput from the queue's results (`pool queue results prompts > results.jsonl`), and cost per 1M tokens."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("results")
parser.add_argument("--standalone", type=float, help="the baseline's tokens/s, for the ratio")
parser.add_argument("--watts", type=float, default=140.0, help="machine power while generating (measured, or the rating)")
parser.add_argument("--price-kwh", type=float, default=0.15)
parser.add_argument("--hardware-usd", type=float, default=3999.0)
parser.add_argument("--years", type=float, default=3.0)
args = parser.parse_args()
rows = [json.loads(line)["result"] for line in open(args.results) if line.strip()]
tokens = sum(r["tokens"] for r in rows)
seconds = max(r["finished"] for r in rows) - min(r["started"] for r in rows)
rate = tokens / seconds
hourly = args.watts / 1000 * args.price_kwh + args.hardware_usd / (args.years * 365 * 24)
out = {"tasks": len(rows), "tokens": tokens, "seconds": round(seconds, 1), "tokens_per_s": round(rate, 1),
       "usd_per_hour": round(hourly, 4), "usd_per_1m_tokens": round(hourly / (rate * 3600) * 1e6, 4)}
if args.standalone:
    out["of_standalone"] = round(rate / args.standalone, 3)
print(json.dumps(out))

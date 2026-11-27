"""1.0 for the right number after "####", 0.1 for the format with a wrong number, else 0."""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    found = re.findall(r"####\s*\$?(-?[\d,]+)", solution_str)
    if not found:
        return 0.0
    return 1.0 if found[-1].replace(",", "") == ground_truth else 0.1

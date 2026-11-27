---
id: every-kind
title: Every kind, in Gym Markdown
---

One exercise of every kind, all written in a single Gym Markdown file: a lesson, choices, typed answers, ordering,
matching, blanks, Python code with tests, a written answer and a checklist.

# Reading and answering
A lesson, then questions the board grades in an instant.

## How this gym is written
minutes: 2

This whole gym is one Markdown file. Each `##` heading is an exercise, and its sections decide what kind it is.

> [!TIP]
> Run `python -m gymhub.cli gym export gyms/ml-fundamentals` to see a full course written this way.

## Binary search cost
id: binary_search_cost
difficulty: easy
minutes: 1
concepts: complexity

How many comparisons does binary search make on a sorted array of $n$ items, in the worst case?

### Hints
- Each comparison halves the range that is left.

### Choices
- [ ] $O(1)$
- [x] $O(\log n)$
- [ ] $O(n)$

### Explanation
Each step halves the range, so about $\log_2 n$ steps.

## Spot the primes
id: primes
difficulty: easy

Select every prime number.

### Choices
- [x] 2
- [ ] 9
- [x] 11
- [ ] 15

## Half of a quarter
id: eighth
difficulty: easy
placeholder: a decimal

What is half of $0.25$?

### Answer
- 0.125 ± 0.0005

### Explanation
$0.25 / 2 = 0.125$.

## From URL to page
id: request_path
difficulty: easy
concepts: networking

You type `https://example.com` and press Enter. Put what the browser does in order, first step at the top.

### Order
1. Resolve the host name with DNS
2. Open a TCP connection
3. Negotiate TLS
4. Send the HTTP request

### Explanation
An address comes before a connection, a connection before encryption, and encryption before the request.

## What each operation costs
id: cost_of_operations
difficulty: easy

Match each operation with its time complexity. One option is not used.

### Match
| Operation | Cost |
|---|---|
| Look up a key in a hash table | $O(1)$ on average |
| Binary search a sorted array | $O(\log n)$ |
| Find the maximum of an unsorted array | $O(n)$ |
|  | $O(n^2)$ |

## A running mean
id: running_mean
difficulty: easy

Fill in each blank.

### Blanks
After the values 2, 4 and 9 the mean is [[5 ± 0.001]]. Adding a value to $n$ values divides by [[n + 1 | n+1 | 1 + n]].

## Sum in a loop
id: sum_loop
difficulty: easy

Complete the loop that sums `xs`.

### Blanks
```python
total = [[0]]
for x in xs:
    total [[+=]] x
```

# Longer work
Code a machine grades, a paragraph graded against a rubric, and a checklist.

## Two sum
id: two_sum
difficulty: easy
minutes: 10
concepts: hashing

Return the indices `[i, j]`, `i < j`, of the two numbers in `nums` that add up to `target`. Exactly one pair does.

### Hints
- For each number, the one you need is `target - x`.
- A dictionary from value to index finds it in one pass.

### Starter
```python
def solve(nums, target):
    pass
```

### Solution
```python
def solve(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```

### Tests
```python
solve([2, 7, 11, 15], 9) == [0, 1]  # example
solve([3, 3], 6) == [0, 1]  # duplicates
solve([-4, 10, 1, 8],
      4)  # over two lines; no ==, so the solution gives the answer
```

### Editorial
Learners see this after their first submission (or when they choose to open it early).

#### Idea
Checking every pair is $O(n^2)$. Walk the list once instead and remember each value's index in a dictionary:
for `x`, the partner is `target - x`, and if it was seen earlier the pair is found.

#### Cost
One pass, $O(n)$ time and $O(n)$ extra memory.

#### Pitfalls
- Look the partner up *before* storing `x`, or `[3, 3]` with target 6 pairs an element with itself.
- Return the earlier index first.

## Why caches help
id: why_caches
difficulty: medium
words: 30-250
pass_score: 0.75

In a short paragraph, explain why a CPU cache makes programs faster, and name one access pattern that makes it much less
effective.

### Rubric
1. Says main memory is much slower than the cache.
2. Mentions locality: recently or nearby accessed data is likely to be used again.
3. Names a cache-unfriendly pattern, such as random access over a large array.

## Your first week
id: first_week
difficulty: easy

Tick each item off when you have done it.

### Checklist
- Solve one exercise in every gym
- Read the hints for an exercise you solved

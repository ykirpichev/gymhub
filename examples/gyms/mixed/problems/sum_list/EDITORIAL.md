#### Idea
Keep a running total and add each number to it. Python's built-in `sum(nums)` does exactly that, and it already
returns 0 for an empty list.

#### Cost
$O(n)$ time, $O(1)$ extra memory.

#### Pitfalls
- Starting the total at `nums[0]` fails on the empty list.

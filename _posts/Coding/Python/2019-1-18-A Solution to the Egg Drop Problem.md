![title image](http://wx4.sinaimg.cn/mw690/006a0Rdhly1fxcjia5tu5j32ds1scnl3.jpg)

### Background

Recently I am reading the book "Cracking the Coding Interview", and in Chapter 6 the egg drop problem caught my attention. The solution shows an intuitive way of thinking, but has no code for how we can generate the result.

### The Egg Drop Problem

There is a building of 100 floors. If an egg drops from the Nth floor or above, it will break. If it's dropped from any floor below, it will not break. Your're given two eggs. Find N, while minimizing the number of drops for the worst case.

### Code

```python
from functools import lru_cache


def numberofdrop(n):
    @lru_cache(None)
    def drop(start, end, dropped):
        if start >= end:
            return 0
        return min(
            max(drop(i + 1, end, dropped + 1), dropped + 1 + i - start)
            for i in range(start, end))

    return drop(0, n, 0)
```

### Usage

numberofdrop(100) will give you the answer 14, which is the number of drops we have to take for the problem with a building of 100 floors.

### Side Notes

1. "start" and "end" define the range of floors still need to be checked.
2. "dropped" is the number of drops we already took.
3. For more explanation, you can check [this blog](http://datagenetics.com/blog/july22012/index.html).

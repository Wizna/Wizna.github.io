![](https://raw.githubusercontent.com/Wizna/play/master/20140526024838215.jpg)

# Background

Well, actually I already has a blog on python cheat sheet, but I found that writing those code in gists make loading time of the page unbearable long. So I decided to put the python notes into plain markdown.

# Code

check whether `x` is a subsequence of `s`

```python 
def isSubsequence(x, s):
    it = iter(s)
    return all(c in it for c in x)
```



# Notes

- Int: in python3, the plain `int` type is unbounded.

## Global interpreter lock (GIL):  

- The mechanism used by the [CPython](https://docs.python.org/3/glossary.html#term-cpython) interpreter to assure that only one thread executes Python [bytecode](https://docs.python.org/3/glossary.html#term-bytecode) at a time. https://docs.python.org/3/glossary.html#term-global-interpreter-lock

## Threading

### Lock

- 有`acquire()` 和`release()`，就是简单的让不让进

### Condition

- `acquire()` and `release()`

### Semaphore

- 可以是`Semaphore(value=3)`，会保证计数$\ge 0$

### Event

- thread间通信，one thread signals an event and other threads wait for it.
- `set()` and `clear()`

### Barrier

- 也是通信，`Barrier(parties=2)`，当有parties这么多个threads同时等待时，一起释放


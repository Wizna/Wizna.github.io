![title image](http://wx3.sinaimg.cn/mw690/006a0Rdhgy1fl1hfkq0jfj30ku0rsq7u.jpg)

### Background

Python is a powerful and concise language and I believe one-liners can show its beauty and help people have a better understanding of its features, especially its built-in functions. For examples, it push you to use list comprehension, reduce(), map(), filter() and lambda function.

However, keep in mind that one-line is not the ultimate goal instead of readability and performance.



### Problem 1

Given an array of integers `A` sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.

**Example**

```json
Input: [-4,-1,0,3,10]
Output: [0,1,9,16,100]
```

**Solution**

```python
def sortedSquares(A):
    """
    :type A: List[int]
    :rtype: List[int]
    """
    return [
        A.pop(0 if abs(A[0]) > abs(A[j]) else j)**2
        for j in range(len(A) - 1, -1, -1)
    ][::-1]
```

**Side Notes**

1. O(n), 2 pointers.



### Problem 2

Write a function to multiply two positive integers `a` and `b` without using the `*` operator (or `/` operator). You can use addition, subtraction, and bit shifting.

**Solution**

```python
def multiply(a, b):
    """
    :type a: int
    :type b: int
    :rtype: int
    """
    return sum(b << i for i in range(32) if a >> i & 1)
```

**Side Notes**

1. Works for 32-bit integers.



### Problem 3

We have a list of `points` on the plane.  Find the `K` closest points to the origin `(0, 0)`.

(Here, the distance between two points on a plane is the Euclidean distance.) 

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)

**Solution**

```python
def kClosest(points, K):
    """
    :type points: List[List[int]]
    :type K: int
    :rtype: List[List[int]]
    """
    return sorted(points, key = lambda x: (x[0]**2 + x[1]**2))[:K]

```



### Problem 4

Given `N`, calculate **Fibonacci number** `F(N)`, starting from `0` and `1`.

**Solution**

```python
from functools import reduce
def fib(N):
    """
    :type N: int
    :rtype: int
    """
    return reduce(lambda x, _: [x[1], sum(x)], range(N), [0, 1])[0]
```



### Problem 5

In an alien language, surprisingly they also use english lowercase letters, but possibly in a different `order`. The `order` of the alphabet is some permutation of lowercase letters.

Given a sequence of `words` written in the alien language, and the `order` of the alphabet, return `true` if and only if the given `words` are sorted lexicographically in this alien language.

**Example**

```json
Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
```

**Solution**

```python
def isAlienSorted(words, order):
    """
    :type words: List[str]
    :type order: str
    :rtype: bool
    """
    return sorted(words, key=lambda x: list(map(order.index, x))) == words
```



### Problem 6

You are given an `n x n `2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

**Note**

You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.

**Solution**

```python
def rotate(matrix):
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    matrix[:] = map(list, zip(*matrix[::-1]))
```



### Problem 7

There is a list of sorted integers from 1 to *n*. Starting from left to right, remove the first number and every other number afterward until you reach the end of the list.

Repeat the previous step again, but this time from right to left, remove the right most number and every other number from the remaining numbers.

We keep repeating the steps again, alternating left to right and right to left, until a single number remains.

Find the last number that remains starting with a list of length *n*.

**Example**

```json
Input:
n = 9,
1 2 3 4 5 6 7 8 9
2 4 6 8
2 6
6

Output:
6
```

**Solution**

```python
def lastRemaining(n):
    """
    :type n: int
    :rtype: int
    """
    return n // 2 + 1 if n <= 3 else lastRemaining(n // 4) * 4 - (n % 4 < 2) * 2
```





### To be continued ...


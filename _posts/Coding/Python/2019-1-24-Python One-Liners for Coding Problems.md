![title image](https://images-na.ssl-images-amazon.com/images/I/51xmxTlKcpL._SX425_.jpg)

### Background

Python is a powerful and concise language and I believe one-liners can show its beauty and help people have a better understanding of its features, especially its built-in functions. For examples, it push you to use list comprehension, reduce(), map(), filter() and lambda function.

However, keep in mind that one-line is not the ultimate goal instead of readability and performance.

### Problem 1

Given an array of integers `A` sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.

**Example:**

```
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
    return reduce(lambda x, _: [x[-1], sum(x)], range(N), [0, 1])[0]
```



### To be continued ...


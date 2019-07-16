![title image](http://wx3.sinaimg.cn/mw690/006a0Rdhgy1fl1hfkq0jfj30ku0rsq7u.jpg)

### Background

Python is a powerful and concise language and I believe one-liners can show its beauty and help people have a better understanding of its features, especially its built-in functions. For examples, it push you to use list comprehension, reduce(), map(), filter() and lambda function.

However, keep in mind that one-line is not the ultimate goal instead of readability and performance.

Please work on the puzzles before you click on "Solution".



### Problem 1

Given an array of integers `A` sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.

**Example**

```json
Input: [-4,-1,0,3,10]
Output: [0,1,9,16,100]
```

<details>
<summary>Solution</summary>
<script src="https://gist.github.com/Wizna/93d18ce47a5beb7d6942294ed07bccb4.js"></script>
</details>

**Side Notes**

1. O(n), 2 pointers.



### Problem 2

Write a function to multiply two positive integers `a` and `b` without using the `*` operator (or `/` operator). You can use addition, subtraction, and bit shifting.

<details><summary>Solution</summary>
<script src="https://gist.github.com/Wizna/0f1b920ed59a240fe06150e6d23a4f50.js"></script>
</details>

**Side Notes**

1. Works for 32-bit integers.



### Problem 3

We have a list of `points` on the plane.  Find the `K` closest points to the origin `(0, 0)`.

(Here, the distance between two points on a plane is the Euclidean distance.) 

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)

<details><summary>Solution</summary>
<script src="https://gist.github.com/Wizna/be97effec90463e902ee9a8267d3cb50.js"></script>
</details>


### Problem 4

Given `N`, calculate **Fibonacci number** `F(N)`, starting from `0` and `1`.

<details><summary>Solution</summary>
<script src="https://gist.github.com/Wizna/c8dd054682cee8472a9a8b40e8b98b50.js"></script>
</details>


### Problem 5

In an alien language, surprisingly they also use english lowercase letters, but possibly in a different `order`. The `order` of the alphabet is some permutation of lowercase letters.

Given a sequence of `words` written in the alien language, and the `order` of the alphabet, return `true` if and only if the given `words` are sorted lexicographically in this alien language.

**Example**

```json
Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
```

<details><summary>Solution</summary>
<script src="https://gist.github.com/Wizna/60a46b741994120c881f2c142eb17bd3.js"></script>
</details>


### Problem 6

You are given an `n x n `2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

**Note**

You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.

<details><summary>Solution</summary>
<script src="https://gist.github.com/Wizna/1f6c6382e5653ffcff43d7d0b5982592.js"></script>
</details>


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

<details><summary>Solution</summary>
<script src="https://gist.github.com/Wizna/606702a3ca194f283fe11a627a4f542f.js"></script>
</details>


### Problem 8

Find the index of `n`th occurrence of substring `b` in string `a`.

<details><summary>Solution</summary>
<script src="https://gist.github.com/Wizna/7b6f9b6c7db7f95b0d4390002b9e660a.js"></script>
</details>

**Side Notes**

1. If finding 4th of `b` in `a` is -1, then obviously the result of 5th should also be -1.
2. `y` is `[0, 1, 2, 3, ...]`, increase 1 at a time, so it always increases slower than x (the indices), so `y > x + 1` is equivalent to `x == -1 and y != 0`.
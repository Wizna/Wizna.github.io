![titile image](https://d3i6fh83elv35t.cloudfront.net/newshour/app/uploads/2016/05/729665main_A-BlackHoleArt-pia16695_full-1024x576.jpg)

### Problem

return the total number of distinct solutions for N-Queens problem.

### Naïve solution

It is easy to come up with a solution using DFS.

The code is as follows:

```python
def getQ(n):
    primary = {i for i in range(n)}
    used = [-1] * n

    def toFit(nums, length):
        re1 = [nums[i] + length - i for i in range(length)]
        re2 = [nums[i] - length + i for i in range(length)]
        return set(re1 + re2 + nums[:length])
    
    def fix(k):
        if k == 0:
            return 1
        dfects = toFit(used, n - k)
        result = 0
        for item in primary - dfects:
            used[n - k] = item
            result += fix(k - 1)
        return result

    return fix(n)
```

Function toFit return all row positions that will result in conflicts for current column.

Function fix() try out each possible row position column by column and search iteratively.

The code is simple and easy to understand, but it takes 1m 6.30s to compute getQ(13).

We need to make it faster.

### Final solution

We notice that checking the validity of potential positions is time consuming, and we can reduce the time by changing how we check.

There are 2n-1 diagonals from top left to bottom right and 2n-1 diagonals from top right to bottom left for a n\*n board. We can use 2 arrays to store all possessed diagonals and check whether the new position is a potential solution by a simple array access.

The code is as follows:

```python
def newtoQ(n):
    """
    :type n: int
    :rtype: int
    """
    primary = {i for i in range(n)}
    used = set()
    cord1 = [False] * (2 * n - 1)
    cord2 = [False] * (2 * n - 1)

    def fix(k):
        if k == 0:
            return 1

        result = 0
        for item in primary - used:
            if not cord1[item + k - 1] and not cord2[item + n - k]:
                used.add(item)
                cord1[item + k - 1] = True
                cord2[item + n - k] = True
                result += fix(k - 1)
                cord1[item + k - 1] = False
                cord2[item + n - k] = False
                used.remove(item)

        return result

    if n == 0:
        return 1
    result = 0
    half = int(n / 2)
    for i in range(half):
        used.add(i)
        cord1[i + n - 1] = True
        cord2[i] = True
        result += fix(n - 1)
        cord1[i + n - 1] = False
        cord2[i] = False
        used.remove(i)

    result *= 2
    if n % 2:
        used.add(half)
        cord1[half + n - 1] = True
        cord2[half] = True
        result += fix(n - 1)
        cord1[half + n - 1] = False
        cord2[half] = False
        used.remove(half)
    return result
```

Now we go through the code.

In line 6, I use a variable primary to avoid regenerating the same set {1..n} again and again, which is time consuming.

In line 8,9, we initialize the 2 arrays for diagonals, False indicating not possessed by any queen.

In line 16, we use a difference of set to filter possible positions, which is checking the rows.

In line 17, we check two directions of diagonals.

In line 28, I add this check condition just to make it consistent to the answers of LeetCode, I personally think when n is 0, there is no solution.

From line 31 to 39, I do a trick to cut the search space. As we all know, the board is symmetric, so the selection of row of the first queen in column 0 can be limited to 0~n/2, then we multiply the result number by 2.

Line 42 to 49, it is possible that n is odd, so there is a case that the 1st queen is put at row int(n/2) in column 0, if that happens, we can do a similar trick as above for column 1. we again cut the search space by half.

So all in all, the trick make the execution time to be half of those without it.

### Result 

The final solution takes 64ms to pass all 9 test cases, and beats 97% of solutions in LeetCode. The computing time of toQ(13) drops to 9.5s. Although it is not beautiful anymore, it is fast.


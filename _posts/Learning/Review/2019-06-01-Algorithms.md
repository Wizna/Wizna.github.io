![title image](https://raw.githubusercontent.com/Wizna/play/master/84299-457783.jpg)

* TOC
{:toc}
 


##背景
简单复习一下最基本的算法和数据结构，可惜一直耽搁了。

部分来自introduction to algorithm和https://github.com/keon/algorithms，还有网络。

## Sorting

### Quicksort

```python
def quicksort(self, nums):
    if len(nums) <= 1:
        return nums

    pivot = random.choice(nums)
    lt = [v for v in nums if v < pivot]
    eq = [v for v in nums if v == pivot]
    gt = [v for v in nums if v > pivot]

    return self.quicksort(lt) + eq + self.quicksort(gt)
```

### Heapsort

- The (binary) heap data structure is an array object that we can view as a nearly complete binary tree (except possibly the lowest)

- sort in place

- runtime $O(n \lg n)$, height $\Theta(\lg  n)$ 

- index of array starts at 1.

```python
  def parent(i):
      return i // 2
  
  def left(i):
      return 2 * i
  
  def right(i):
      return 2 * i + 1

  def max_heapify(A, i):
      l = left(i)
      r = right(i)
      if l <= A.heap_size and A[l] > A[i]:
          largest = l
      else:
          largest = i
      if r <= A.heap_size and A[r] > A[largest]:
          largest = r
      if largest != i:
          A[i], A[largest] = A[largest], A[i]
          max_heapify(A, largest)
          
          
  def build_max_heap(A):
      A.heap_size = len(A)
      for i in range(len(A) // 2, 0, -1):
          max_heapfiy(A, i)
```


## Data Structures

### Red-Black Trees

### AVL Trees

### Augmenting data structures

### B-Trees

### Disjoint sets (Union find)

- 

### Trie



### Segment Tree

### Linkedlist

- to find if there is a cycle in a linked list 

```python 
  class Node:
      def __init__(self, x):
          self.val = x
          self.next = None
  
  def is_cyclic(head):
      """
      :type head: Node
      :rtype: bool
      """
      if not head:
          return False
      runner = head
      walker = head
      while runner.next and runner.next.next:
          runner = runner.next.next
          walker = walker.next
          if runner == walker:
              return True
      return False
```

## Graph

### Topological sort 

### Minimum spanning tree

### Single-Source Shortest Paths

### All-Pairs Shortest Paths 

### Maximum flow

## Selected topics

### Linear programming

### String matching

#### Rabin-Karp



### Computational geometry

## Miscellaneous

### Parition of a number

- **partition** of a positive [integer](https://en.wikipedia.org/wiki/Integer) *n*, also called an **integer partition**, is a way of writing *n* as a [sum](https://en.wikipedia.org/wiki/Summation) of positive integers，不关心顺序

```python
  # 这个是输出所有的partition结果
  def partitions(n, I=1):
      yield (n, )
      for i in range(I, n // 2 + 1):
          for p in partitions(n - i, i):
              yield (i, ) + p
              
  # 这个是计算有多少种分法，其实和上面的类似            
  def par(n, b=1):
      r = 1
      for i in range(b, n // 2 + 1):
          r += par(n - i, i)
      return r
```

### Permutations & combinations

### Selection rank 

- similar to quick sort



### Newton-Raphson algorithm

### PSO algorithm

### Fenwick tree





# To be continued ...





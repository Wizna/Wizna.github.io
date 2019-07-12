![title image](https://a.ksd-i.com/a/2016-09-17/84299-457783.jpg)



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

- runtime $O(n \lg n)$, height $\theta(\lg  n)$ 

- index of array starts at 1.

- ```python
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

- 



## Data Structures

### Red-Black Trees

### Augmenting data structures

### B-Trees

### Disjoint sets

### Trie

### Segment Tree



## Graph

### Topological sort 

### Minimum spanning tree

### Single-Source Shortest Paths

### All-Pairs Shortest Paths 

### Maximum flow

## Selected topics

### Linear programming

### String matching

### Computational geometry

## Miscellaneous

### Permutations & combinations

### Selection rank 

- similar to quick sort
- 



# To be continued ...





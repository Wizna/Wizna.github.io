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

### Traverse

```python
# 本质就是保存一个visited，然后遍历，dfs用stack，bfs用queue
def dfs_traverse(graph, start):
    visited, stack = set(), [start]
    while stack:
        node = stack.pop()
        visited.add(node)
        for nextNode in graph[node]:
            if nextNode not in visited:
                stack.append(nextNode)
    return visited
```



### Topological sort 

- 就是按照图中顺序，得到一个排序，DAG都有至少一个解，有环就没有解，一般的复杂度都是$O(|V|+|E|)$

```python
# Kahn's algorithm
def topo_sort(points, pre: Dict[str, set], suc: Dict[str, set]):
    order = []
    sources = [p for p in points if not pre[p]]
    while sources:
        s = sources.pop()
        order.append(s)
        for u in suc[s]:
            pre[u].remove(s)
            if not pre[u]:
                sources.append(u)
    return order if len(order) == len(points) else []


# 一个变种，calculate max latencies
# [('A', 'B', 100), ('A', 'C', 200), ('A', 'F', 100), ('B', 'D', 100),
#            ('D', 'E', 50), ('C', 'G', 300)]
# result = {'A': 500, 'B': 150, 'C': 300, 'D': 50, 'E': 0, 'F': 0, 'G': 0}
def topo_sort(latencies):
    source = set(i[1] for i in latencies)
    end = set(i[0] for i in latencies)

    next_nodes = defaultdict(set)
    prev_nodes = defaultdict(set)
    time_dict = dict()
    for n, prev, time in latencies:
        next_nodes[prev].add(n)
        prev_nodes[n].add(prev)
        time_dict[(n, prev)] = time

    node_results = defaultdict(int)
    to_decide = source | end

    topo_que = list(source - end)
    while topo_que:
        node = topo_que.pop()
        for v in next_nodes[node]:
            prev_nodes[v].remove(node)
            node_results[v] = max(node_results[node] + time_dict[(v, node)],
                                  node_results[v])
            if not prev_nodes[v]:
                topo_que.append(v)

    return node_results
```


### Minimum spanning tree

### Single-Source Shortest Paths

- Dijkstra’s algorithm用于directed graph，而且edge weights $>= 0$ 
- 复杂度$O(E\log E+V)$

```python 
import heapq


def calculate_distances(graph, starting_vertex):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[starting_vertex] = 0

    pq = [(0, starting_vertex)]
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # Only consider this new path if it's better than any path we've
            # already found.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

example_graph = {
    'U': {'V': 2, 'W': 5, 'X': 1},
    'V': {'U': 2, 'X': 2, 'W': 3},
    'W': {'V': 3, 'U': 5, 'X': 3, 'Y': 1, 'Z': 5},
    'X': {'U': 1, 'V': 2, 'W': 3, 'Y': 1},
    'Y': {'X': 1, 'W': 1, 'Z': 1},
    'Z': {'W': 5, 'Y': 1},
}
print(calculate_distances(example_graph, 'X'))
# => {'U': 1, 'W': 2, 'V': 2, 'Y': 1, 'X': 0, 'Z': 2}
```



### All-Pairs Shortest Paths 

### Maximum flow

- 一个连通图，有s，t两个特殊的节点，其他节点都在s到t的某条path上
- **max-flow min-cut theorem** states that in a [flow network](https://en.wikipedia.org/wiki/Flow_network), the maximum amount of flow passing from the [*source*](https://en.wikipedia.org/wiki/Glossary_of_graph_theory#Direction) to the [*sink*](https://en.wikipedia.org/wiki/Glossary_of_graph_theory#Direction) is equal to the total weight of the edges in the [minimum cut](https://en.wikipedia.org/wiki/Minimum_cut), i.e. the smallest total weight of the edges which if removed would disconnect the source from the sink.
- cut就是去掉一些edges，把s和t分开成两个图
- capacity就是从s侧到t侧的容量和，注意不包括反向从t->s的那些edges的流量
- max flow算法本质就是开始都是0，发现某path的最小处>0，那就加上这么小的流量填满。找path用dfs就是Ford-Fulkerson，用bfs就是 [Edmonds–Karp](https://en.wikipedia.org/wiki/Edmonds–Karp_algorithm)

```python 
"""
Input : capacity, source, sink
Output : maximum flow from source to sink
Capacity is a two-dimensional array that is v*v.
capacity[i][j] implies the capacity of the edge from i to j.
If there is no edge from i to j, capacity[i][j] == 0.
"""
# current_flow刚开始设置很大的值，不断地随着dfs而变小
def dfs(capacity, flow, visit, vertices, idx, sink, current_flow = 1 << 63):
    # DFS function for ford_fulkerson algorithm.
    if idx == sink: 
        return current_flow
    visit[idx] = True
    for nxt in range(vertices):
        if not visit[nxt] and flow[idx][nxt] < capacity[idx][nxt]:
            tmp = dfs(capacity, flow, visit, vertices, nxt, sink, min(current_flow, capacity[idx][nxt]-flow[idx][nxt]))
            if tmp:
                flow[idx][nxt] += tmp
                flow[nxt][idx] -= tmp
                return tmp
    return 0

def ford_fulkerson(capacity, source, sink):
    # Computes maximum flow from source to sink using DFS. Greedy
    # Time Complexity : O(Ef) 也就是和edge还有max flow正比
    # E is the number of edges and f is the maximum flow in the graph.
    vertices = len(capacity)
    ret = 0
    flow = [[0]*vertices for i in range(vertices)]
    while True:
        visit = [False for i in range(vertices)]
        tmp = dfs(capacity, flow, visit, vertices, source, sink)
        if tmp: 
            ret += tmp
        else: 
            break
    return ret

```




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





![titile image](https://images.pexels.com/photos/459225/pexels-photo-459225.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)

### Background: 

I have been using Python intensively for about 1 year. From now and then I look up some of the common operations and I decided to record all the code snippets in a handbook.



### Table:

{:class="table table-bordered"}
| **No.** | **Goal**                                                     | **Solution**                                                 | **Note**                                                     |
| :-----: | :----------------------------------------------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
|    1    | Reverse a string or a list                                   | <script src="https://gist.github.com/Wizna/6d8bc2a47af1ce6a83af4e5227eb9e00.js"></script> |                                                              |
|    2    | Join a list of strings to a single string                    | <script src="https://gist.github.com/Wizna/2a9441980881bbe6b48c33209fe7b707.js"></script> |                                                              |
|    3    | Flatten list from list of list                               | <script src="https://gist.github.com/Wizna/dc4030c27711837cb9a25b7e8947a350.js"></script> |                                                              |
|    4    | Trim a string                                                | mystr.strip()                                                |                                                              |
|    5    | String replace a substring pattern with another substring    | mystr.replace('old', 'new')                                  |                                                              |
|    6    | Cast  float or string to int                                 | int(mystr)                                                   | auto floor to the nearest integer                            |
|    7    | Check string or char is in uppercase                         | mystr.isupper()                                              |                                                              |
|    8    | If statement in list comprehension                           | [value for index,value in enumerate(mylist) if value != 0]   |                                                              |
|    9    | If else statement in list comprehension                      | row = [None, '\u0303', '\u1200', '\u0203']<br>w = [x if x is not None else '' for x in row] |                                                              |
|   10    | Transpose matrix  (2-d array)                                | nmatrix = [list(i) for i in zip(*matrix)]                    |                                                              |
|   11    | Rotate matrix (2-d array) 90° clockwise                      | nmatrix = [list(i) for i in zip(*matrix[::-1])]              | if you want to do it inplace, do:<br> matrix[:] = [list(i) for i in zip(*matrix[::-1])] |
|   12    | Check whether str1 is substring of str2                      | str1 in str2                                                 |                                                              |
|   13    | Get Current work directory                                   | import os<br>cwd = os.getcwd()                               |                                                              |
|   14    | Iterate dictionary                                           | for key, value in mydict.items(): <br>...                    |                                                              |
|   15    | Split string with multiple delimiters                        | import re <br>re.split(';\|,\|\*\|\n', 'He,llo * worl;d')    |                                                              |
|   16    | Merge 2 dictionaries                                         | from collections import Counter<br/>a = Counter({'menu': 20, 'good': 15, 'happy': 10, 'bar': 5})<br/>b = Counter({'menu': 1, 'good': 1, 'bar': 3})<br/>a.update(b) | new value is the sum of 2 values from 2 dictionaries         |
|   17    | Initialize list with same primitive values                   | mylist = [True] * 400                                        | don't use this with 2-d array                                |
|   18    | Python supports tertiary  operator                           | if 3 < 5 < 8: print(5)                                       |                                                              |
|   19    | Sort a list of strings by length                             | mylist.sort(key = len)                                       |                                                              |
|   20    | String split by keep the delimiters                          | import re<br>re.split('(;\|,\|\*\|\n)', 'He,llo * worl;d')   | just add outer ( ) to the delimiters                         |
|   21    | Sort dictionary by values                                    | import operator<br>x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}<br>sorted_x = sorted(x.items(), key=operator.itemgetter(1)) | if want to sort by keys, change itemgetter(1) to itemgetter(0) |
|   22    | print 2-d array in a better format                           | import numpy<br>np.matrix(mymatrix)                          |                                                              |
|   23    | Find index and last index of char in string                  | mystr.index(c)<br>mystr.rindex(c)                            |                                                              |
|   24    | Find index and last index of item in list                    | mylist.index(item)<br>len(mylist) - 1 - mylist[::-1].index(item) |                                                              |
|   25    | automatically initialize for new key in dictionary           | from collections import defaultdict <br>mydict = defaultdict(int) | input of defaultdict should be callable, and takes no arguments |
|   26    | Get arbitrary one element from set                           | next(iter(myset))                                            |                                                              |
|   27    | Remove element at index 2 from the list                      | mylist.pop(2)                                                | if no argument, last one is removed                          |
|   28    | Find locations (start, end) of matches with regular expression | myiter = re.finditer('[A-Z][a-z]*', 'This Is My House')<br>indices = [m.span() for m in myiter] |                                                              |
|   29    | Update each value in a dictionary with an operation          | mydict.update((x, y*2) for x, y in mydict.items())           |                                                              |
|   31    | Find all indices of an element in a list                     | indices = [i for i, x in enumerate(mylist) if x == "whatever"] |                                                              |
|   32    | Return a random element from a list                          | import random<br>random.choice(mylist)                       |                                                              |
|   33    | Product of all elements of a list                            | from operator import mul<br>from functools import reduce<br>reduce(mul, [1, 2, 3, 4, 5, 10], 1) |                                                              |
|   34    | lexicographical  compare of strings                          | stra > strb <br>stra <= strb                                 | "abcd" > "ab" and "ad" > "ab"                                |
|   35    | integer division                                             | 7 // 3                                                       | round down                                                   |
|   36    | two's complement                                             | ~1314                                                        | equals to -1315                                              |
|   37    | padding spaces to a string to a certain length               | mystr.center(30)<br>mystr.ljust(10)<br>mystr.rjust(3)        |                                                              |
|   38    | Get a random float                                           | random.random()                                              |                                                              |
|   39    | Get a random integer from range                              | random.randrange(6)                                          |                                                              |
|   40    | Get profile of<br/>program, print out most time consuming function calls | import cProfile<br/><br/>cProfile.run('int(13)')             |                                                              |
|   41    | Get<br/>line by line time execution in jupyter notebook      | %load_ext line_profiler<br/>%lprun -f totalNQueens totalNQueens(13) | need to specify the function name (e.g. totalNQueens) and the real execution call (e.g. totalNQueens(13))<br/>should use "" when passing strings as arguments |
|   42    | Usage<br/>of doctest                                         | def average(values):<br/>    """<br/>    >>> average([20, 30, 70])<br/>    40.0<br/>    """<br/>    return sum(values) / len(values)<br/>import doctest<br/>doctest.testmod() | there must be at least 1 space on the right side of >>>      |
|   43    | Double<br/>ended queue                                       | from collections import deque<br/>d = deque('ghi')           | can be think of a queue + stack                              |
|   44    | Print<br/>out nicely                                         | from pprint import pprint<br/>pprint(….)                     |                                                              |
|   45    | Permutation<br/>and combination                              | from itertools import combinations<br/><br/>list(combinations([1, 2, 3, 4], 2)) | output: [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]     |
|   46    | Get<br/>all attributes and their values of an object         | vars(myobject)                                               |                                                              |
|   47    | Get the<br/>representation string of the object              | repr(myobject)                                               | if not specified, it will be like:<br/>'<__main__.Duck object at 0x00000133BDE7A6D8>' |
|   48    | Save<br/>memory by using generator instead of list           | Use <br/>sum(x*x for x in range(10))                         | instead of <br/>sum([x*x for x in range(10)])                |
|   49    | variables<br/>created inside a loop are accessible outside the loop scope | for j in mylist:<br/>    print(j)<br/>print(j)               | the<br/>j actually is assigned the last element of the list mylist |
|   50    | check whether a list iterate till end normally               | for item in mylist: <br/>    ...<br/>    break<br/>else: <br/>    Do things that should be done if mylist is thoroughly traversed |                                                              |
|   51    | to<br/>make class objects comparable, just add function \_\_lt\_\_ | def \_\_lt\_\_(self, other):<br/>        return ((self.last, self.first) < (other.last, other.first)) |                                                              |
|   52    | to<br/>make class objects hashable                           | Implement \_\_hash\_\_<br/>Also need to take care of \_\_eq\_\_ and make it compatible with hash |                                                              |



### To be continued ...
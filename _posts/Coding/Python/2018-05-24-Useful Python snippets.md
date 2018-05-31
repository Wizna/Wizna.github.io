![titile image](https://images.pexels.com/photos/459225/pexels-photo-459225.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)

# Useful Python snippets

### Background: 

I have been using Python intensively for about 1 year. From now and then I look up some of the common operations and I decided to record all the code snippets in a handbook.



### Table:

| No.  | Goal                                                         | Solution                                                     | Note                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Reverse a string or a list                                   | 'hello world' [::-1]                                         |                                                              |
| 2    | Join a list of strings to a single string                    | result = ' '.join(mylist)                                    |                                                              |
| 3    | Flatten list from list of list                               | flat_list = [item for sublist in mylist for item in sublist] |                                                              |
| 4    | Trim a string                                                | mystr.strip()                                                |                                                              |
| 5    | String replace a substring pattern with another substring    | mystr.replace('old', 'new')                                  |                                                              |
| 6    | Cast  float or string to int                                 | int(mystr)                                                   | auto floor to the nearest integer                            |
| 7    | Check string or char is upper                                | mystr.isupper()                                              |                                                              |
| 8    | If statement in list comprehension                           | [value for index,value in enumerate(mylist) if value != 0]   |                                                              |
| 9    | If else statement in list comprehension                      | row = [None, '\u0303', '\u1200', '\u0203'] <br>w = [x if x is not None else '' for x in row] |                                                              |
| 10   | Transpose matrix  (2-d array)                                | nmatrix = [list(i) for i in zip(*matrix)]                    |                                                              |
| 11   | Rotate matrix (2-d array) 90 degrees clockwise               | nmatrix = [list(i) for i in zip(*matrix[::-1])]              | if you want to do it inplace, do:<br> matrix[:] = [list(i) for i in zip(*matrix[::-1])] |
| 12   | Check whether str1 is substring of str2                      | str1 in str2                                                 |                                                              |
| 13   | Get Current work directory                                   | import os<br>cwd = os.getcwd()                               |                                                              |
| 14   | Iterate dictionary                                           | for index, value in mydict.items(): <br>...                  |                                                              |
| 15   | Split string with multiple delimiters                        | import re <br>re.split(';\|,\|\*\|\n', 'He,llo * worl;d')    |                                                              |
| 16   | Merge 2 dictionaries                                         | result = {k: resultDictL.get(k, 0) + resultDictR.get(k, 0) for k in set(resultDictL) \| set(resultDictR)} |                                                              |
| 17   | Initialize list with same primitive values                   | mylist = [True] * 400                                        | don't use this with 2-d array                                |
| 18   | Python supports tertiary  operator                           | if 3 < 5 < 8: print(5)                                       |                                                              |
| 19   | Sort a list of strings by length                             | mylist.sort(key = len)                                       |                                                              |
| 20   | String split by keep the delimiters                          | import re<br>re.split('(;\|,\|\*\|\n)', 'He,llo * worl;d')   | just add outer ( ) to the delimiters                         |
| 21   | Sort dictionary by values                                    | import operator<br>x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}<br>sorted_x = sorted(x.items(), key=operator.itemgetter(1)) | if want to sort by keys, change itemgetter(1) to itemgetter(0) |
| 22   | print 2-d array in a better format                           | import numpy<br>np.matrix(mymatrix)                          |                                                              |
| 23   | Find index and last index of char in string                  | mystr.index(c)<br>mystr.rindex(c)                            |                                                              |
| 24   | Find index and last index of item in list                    | mylist.index(item)<br>len(mylist) - 1 - mylist[::-1].index(item) |                                                              |
| 25   | automatically initialize for new key in dictionary           | from collections import defaultdict <br>mydict = defaultdict(int) | input of defaultdict should be callable, and takes no arguments |
| 26   | Get arbitrary one element from set                           | next(iter(myset))                                            |                                                              |
| 27   | Remove element at index 2 from the list                      | mylist.pop(2)                                                | if no argument, last one is removed                          |
| 28   | Find locations (start, end) of matches with regular expression | myiter = re.finditer('[A-Z][a-z]*', 'This Is My House')<br>indices = [m.span() for m in myiter] |                                                              |
| 29   |                                                              |                                                              |                                                              |
| 30   |                                                              |                                                              |                                                              |
| 31   |                                                              |                                                              |                                                              |
| 32   |                                                              |                                                              |                                                              |
| 33   |                                                              |                                                              |                                                              |
| 34   |                                                              |                                                              |                                                              |
| 35   |                                                              |                                                              |                                                              |
| 36   |                                                              |                                                              |                                                              |
| 37   |                                                              |                                                              |                                                              |
| 38   |                                                              |                                                              |                                                              |
| 39   |                                                              |                                                              |                                                              |


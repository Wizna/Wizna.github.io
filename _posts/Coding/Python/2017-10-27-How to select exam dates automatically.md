![image spanner](https://images.freecreatives.com/wp-content/uploads/2015/06/beautiful-nature-scenery.jpg)

### Background

In my major Software Systems Engineering here in RWTH, often there are 2 exam dates for each course and one can take either of them. It's always annoying for me to select the dates, maybe I have a little bit of select dyslexia.

### Code

```python
from datetime import date
import datetime
import operator
import math

def countDays(selectList):
    selectList.sort()
    now = datetime.datetime.now()
    year = now.year
    if now.month >= 9:
        year += 1
    days = 0
    for i in range(len(selectList) - 1):
        d0 = date(year, selectList[i][0], selectList[i][1])
        d1 = date(year, selectList[i + 1][0], selectList[i + 1][1])
        delta = d1 - d0
        days += math.log2(delta.days+1)
    return days


def selection(exams):
    selectDict = {}
    num = len(exams)
    for selection in range(2**num):
        selectList = []
        for i in range(num):
            index = (selection >> i) & (len(exams[i]) - 1)
            selectList.append(exams[i][index])
        days = countDays(selectList)
        selectDict[str(selectList)] = days
    sortedDict = sorted(selectDict.items(), key=operator.itemgetter(1))[::-1]
    print(sortedDict[0])
    return sortedDict
```

### Usage

Sample Input:

`
selection([[[2,27], [3, 23]], [[2, 26], [3, 19]], [[2,14]], [[3, 1]],[[2,9],[3,20]]])
`

Sample output:

`
('[[2, 9], [2, 14], [3, 1], [3, 19], [3, 23]]', 13.154818109052103)
`

### Notes

1. Input is a list of exam dates.
2. The 2 exam dates for a single course is in a sublist.
3. If there is course which has only 1 exam date, you should also add it into the list as a sublist, since it may influence the result.
4. The idea is to make time span between each pair of consecutive exams longer, and thus give you more time to prepare for each exam.
5. I used logarithm to calculate the score of time span, because preparing an exam for 2 days is far more intense than 6 days, so changing from 2 to 3 should gain more scores than from 6 to 7.
6. The output are the chosen exam dates, and the respective score of this selection.
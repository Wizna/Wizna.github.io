<img src="http://s1.picswalls.com/wallpapers/2016/03/29/beautiful-nature-hd-wallpaper_042322367_304.jpg" />

### Background

This summer semester I selected the practical lab: data analysis and visualization. Probably because of mistake of the teacher assistant, the table of courses' names provided contains garbled texts. They should have save it in utf8 so as to capture those German special characters.



### Idea

This is a simple spell checker on German words with special character. 

It is specially implemented for replacing course' names having '?' with the right character.

I simply followed the Idea of Novig and modified his code.



### Code

```python
import re
from collections import Counter
import pickle

WORDS = pickle.load( open( "germanwords.p", "rb" ) )

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'öäüÖÄÜßáóúéÁÓÚÉ'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    return set(replaces)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correctionWord(word): 
    "Most probable spelling correction for word."
    return max(candidates(word.lower()), key=P)

def correctionStr(string):
    "Correction for a string with multiple words"
    trimstr = re.sub("\s*\(.*\)\s*", "", string)
    strarray = re.findall(r'[A-Z\?]?[a-z\?]+', trimstr)
    for i, substring in enumerate(strarray):
        if "?" in substring:
            strarray[i] = correctionWord(substring)
    resultstr = " ".join(x for x in strarray)        
    return resultstr
```



### Usage

Simply run with: `myCorrectStr = correctionStr("Einf?hrung f?r settings?")`

Then `myCorrectStr` should be `"einführung für settings?"`


### Note

Remember to put the germanwords.p file in the right directory.

I constructed this file by counting on a text file containing 5 German books and 5 semesters of LMU's course schedules.



### Link
[Novig's blog](http://norvig.com/spell-correct.html)

[coursetable.p](https://github.com/Wizna/play/blob/master/coursetable.p)


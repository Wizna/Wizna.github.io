![]()

# Background

Well, actually I already has a blog on python cheat sheet, but I found that writing those code in gists make loading time of the page unbearable long. So I decided to put the python notes into plain markdown.

# Code

check whether `x` is a subsequence of `s`

```python 
def isSubsequence(x, s):
    it = iter(s)
    return all(c in it for c in x)
```



# Notes




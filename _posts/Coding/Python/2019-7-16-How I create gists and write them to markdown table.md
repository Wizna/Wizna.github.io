![title image](http://wx4.sinaimg.cn/mw690/006a0Rdhly1g3lpobkawcj31sg1ccx6r.jpg)

### Background

I have hundreds of code snippets in OneNote, I want to put them into `python cheat sheat.md` file. I achieve this by creating gists and write them into my markdown file with Python. You can see the result at [my blog](https://wizna.github.io/posts/Coding/Python/2018-5-24-Python cheat sheet/).

### Idea

Create a gist for each code snippet, then embed the `<script>` tag into markdown file.

First we need to generate a new personal token [here](https://github.com/settings/tokens).

We only need the permission that relate to gist.

Then we move on to coding.

### Code

```python
import requests
gist_content = "import operator\ns = [[1, 2], [1, 0], [2, 0]]\ns.sort(key=lambda x: (x[0], x[1]))"
gist_name = "create gist example.py"

new_gist = {
    "description": "[Hello] #python",
    "public": "true",
    "files": {
        gist_name: {
            "content": gist_content
        }
    }
}

response = requests.post(
    'https://api.github.com/gists',
    json=new_gist,
    # wizna is my GitHub username, change it to yours
    auth=('wizna', 'YOUR PERSONAL TOKEN STRING'))

gist_id = response.json()['id']
gist_address = '<script src="https://gist.github.com/Wizna/%s.js"></script>' % gist_id
print(gist_address)
```

### Output

`<script src="https://gist.github.com/Wizna/0a08f9881faae58f482812ffa1cf1a45.js"></script>`


<script src="https://gist.github.com/Wizna/0a08f9881faae58f482812ffa1cf1a45.js"></script>

![title image](https://theluxurytravelexpert.files.wordpress.com/2014/01/scenery.jpg)

### Motivation

I did not find any useful directory page for static sites like Jekyll. And I definitely find listing all blogs In chronological order is not handy, so I decided to implement one.

### Idea

First generate html page from structures of the folders of blogs. This is possible since Jekyll supports iterating through subfolders and finding all posts now. Then use JavaScript to modify html page to what I want.

### Result

You can find the demo [here](https://wizna.github.io/directory/). I use jQuery and Bootstrap to ease my coding, so don't forget to import them.

### Code

Only three files were created for the directory, one html file, one JavaScript file and one CSS file.

Reading the posts in the folder _posts and put the path of them in the html page:

```html
<div id="loadfiles">

    {% for post in site.posts %}
    	<span class="post-direct">{{ post.url }}</span> 
    {% endfor %}

</div>
```



We will get something like:

```html
<div id="loadfiles">
    
    	<span class="post-direct">/posts/Learning/RWTH/semester1/2017-4-6-Test4/</span> 
    
    	<span class="post-direct">/posts/Coding/Python/2017-4-6-Test4/</span> 
    
    	<span class="post-direct">/posts/Coding/Python/2017-4-6-Test4%20-%20Copy/</span> 
    
    	<span class="post-direct">/posts/Coding/Python/2017-4-6-Test4%20-%20Copy%20(2)/</span> 
    
</div>
```

The paths are shown in `<span>`s, the last element is the filename while the others are the subfolders' name. Spaces in the filename and folder name are changed into %20.

Note that empty subfolders in the _posts will not show up, since Jekyll only iterates through md files.

Then we use JavaScript to modify the page when document is ready, the first step is to construct a tree from the URL of posts:

```javascript
function getDirectoryStructure() {
    var collection = [];
    var tree = new Tree();
    tree.add('0posts');

    $(".post-direct").each(function(i, obj) {
        var path = $(this).text();
        path = path.substring(1, path.length - 1);
        var strArray = path.split('/');
        for (i = 1; i < strArray.length; i++) {
            if (i == (strArray.length - 1)) {
                tree.add(i + path, (i - 1) + strArray[i - 1]);
            } else {
                //since add to tree does not check multiples, so may result in many duplicate nodes
                if (!collection.includes(i + strArray[i])) {
                    tree.add(i + strArray[i], (i - 1) + strArray[i - 1]);
                    collection.push(i + strArray[i]);
                }
            }
        }
    });

    return tree;
}
```



I did not implement the tree myself, just using a piece of [code](https://github.com/benoitvallon/computer-science-in-javascript/blob/master/data-structures-in-javascript/tree.js) from GitHub.

However, since I have to visit the nodes in a recursive way, so I can't use BFS or DFS provided by the tree structure. But visit the nodes as below:

```javascript
function recurseTree(node) {

    var out = "";
    if (node) {
        if (node.children.length == 0) {
            var strArray = node.data.split('/');
            out = '<li><a href=\"../' + node.data.substring(1) + '\">' + decodeURI(strArray[strArray.length - 1]).replace(/[0-9]+-[0-9]+-[0-9]+-/g, "") + '</a></li>';
        } else {
            for (var i = 0; i < node.children.length; i++) {
                out += recurseTree(node.children[i]);
            }
            if (node.data.substring(1) != 'posts') {
                out = '<li class=\"indent-li\"><label class=\"tree-toggler nav-header\">' + decodeURI(node.data.substring(1)) + '</label><ul class=\"nav nav-list tree\">' + out + '</ul></li>';
            }
        }
    }
    
    return out;

}
```

### A little refinement

Since the visitors can notice changing of content of html, I added a loader to the page.

It is a simple div in html page:

`<div id="loader"></div>`

The corresponding CSS is :

```css
#loader {
    position: absolute;
    left: 50%;
    top: 50%;
    z-index: 1;
    margin: -40px 0 0 -40px;
    border: 16px solid #f3f3f3;
    border-radius: 50%;
    border-top: 16px solid #3498db;
    width: 80px;
    height: 80px;
    -webkit-animation: spin 0.8s linear infinite;
    animation: spin 0.8s linear infinite;
}

@-webkit-keyframes spin {
    0% {
        -webkit-transform: rotate(0deg);
    }
    100% {
        -webkit-transform: rotate(360deg);
    }
}

@keyframes spin {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
```

Add CSS style and class to the `<div>` for path:

```html
<div id="loadfiles" style="display:none;" class="animate-bottom">
    {% for post in site.posts %}
    	<span class="post-direct">{{ post.url }}</span> 
    {% endfor %}
</div>
```



Note that when you want to change the size of the loader, change "margin-top" and "margin-left" to half of its size, so as to make the loader exactly in the center of the screen.

Also, do not forget to change the visibility of the `<div>`s afterwards:

```javascript
function showPage() {

    document.getElementById("loader").style.display = "none";
    document.getElementById("placeHolder").style.display = "none";
    document.getElementById("loadfiles").style.display = "block";

}
```



### Links

[GitHub repository](https://github.com/Wizna/Wizna.github.io)

[directory.html](https://github.com/Wizna/Wizna.github.io/blob/master/directory.html)

[directory.js](https://github.com/Wizna/Wizna.github.io/blob/master/bootstrap/js/directory.js)

[directory.css](https://github.com/Wizna/Wizna.github.io/blob/master/bootstrap/css/directory.css)
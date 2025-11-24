var directoryTitles = {};

$(document).ready(function() {
    if (typeof $.fn.tipuesearch === 'function') {
        $('#tipue_search_input').tipuesearch();
    }

    var windowLoc = $(location).attr('pathname') || '';
    if (!/\/directory\/(index\.html)?$/.test(windowLoc)) {
        return;
    }
    var tree = getDirectoryStructure();
    var renderResult = buildList(tree.root ? tree.root.children : [], true);
    var $directoryTree = $("#directory-tree");

    if (renderResult.count) {
        $directoryTree.html('<div class="directory-card">' + renderResult.html + '</div>');
    } else {
        $directoryTree.html('<p class="directory-empty">暂无文章，敬请期待。</p>');
    }

    var $count = $("#directory-count");
    if ($count.length) {
        $count.text(renderResult.count || 0);
    }

    showPage();
});

function showPage() {
    document.getElementById("loader").style.display = "none";
    document.getElementById("placeHolder").style.display = "none";
    document.getElementById("loadfiles").style.display = "block";
}

function buildList(children, isRoot) {
    if (!children || !children.length) {
        return { html: '', count: 0 };
    }

    var html = '<ul class="directory-list' + (isRoot ? ' directory-root' : '') + '">';
    var total = 0;
    var sorted = children.slice().sort(function(a, b) {
        return getSortKey(a).localeCompare(getSortKey(b));
    });

    for (var idx = 0; idx < sorted.length; idx++) {
        var childResult = buildNode(sorted[idx]);
        if (!childResult.count) {
            continue;
        }
        html += childResult.html;
        total += childResult.count;
    }

    html += '</ul>';

    if (!total) {
        return { html: '', count: 0 };
    }

    return { html: html, count: total };
}

function buildNode(node) {
    if (!node) {
        return { html: '', count: 0 };
    }

    if (!node.children.length) {
        var path = node.data.substring(1);
        var label = formatPostTitle(path);
        var url = '../' + path;

        return {
            html: '<li class="directory-item directory-post-item"><a href="' + url + '" class="directory-link">' + label + '</a></li>',
            count: 1
        };
    }

    var listResult = buildList(node.children, false);
    if (!listResult.count) {
        return { html: '', count: 0 };
    }

    var collapseId = 'collapse-' + sanitizeId(node.data);
    var displayName = formatCategoryTitle(node);

    var html = ''
        + '<li class="directory-item directory-category">'
        + '<button class="category-toggle" type="button" data-toggle="collapse" data-target="#' + collapseId + '" aria-expanded="true">'
        + '<span class="category-name">' + displayName + '</span>'
        + '<span class="category-count">' + listResult.count + '</span>'
        + '</button>'
        + '<div class="collapse in directory-sublist" id="' + collapseId + '">'
        + listResult.html
        + '</div>'
        + '</li>';

    return { html: html, count: listResult.count };
}

function formatCategoryTitle(node) {
    return decodeURIComponent(node.data.substring(1)).replace(/-/g, ' ');
}

function formatPostTitle(path) {
    if (directoryTitles[path]) {
        return directoryTitles[path];
    }

    var segments = path.split('/');
    var slug = decodeURIComponent(segments[segments.length - 1] || '');
    slug = slug.replace(/^\d{4}-\d{2}-\d{2}-/, '');
    slug = slug.replace(/-/g, ' ');
    slug = slug.replace(/\s+/g, ' ').trim();
    return slug || decodeURIComponent(path);
}

function getSortKey(node) {
    if (!node) {
        return '';
    }

    if (!node.children.length) {
        var path = node.data.substring(1);
        var segments = path.split('/');
        var slug = decodeURIComponent(segments[segments.length - 1] || '');
        if (directoryTitles[path]) {
            return directoryTitles[path].toLowerCase();
        }
        slug = slug.replace(/^\d{4}-\d{2}-\d{2}-/, '');
        return slug.toLowerCase();
    }

    return decodeURIComponent(node.data.substring(1)).toLowerCase();
}

function sanitizeId(value) {
    return value.replace(/[^a-zA-Z0-9]+/g, '-').toLowerCase();
}

function getDirectoryStructure() {
    var collection = [];
    var tree = new Tree();
    tree.add('0posts');
    directoryTitles = {};

    $(".post-direct").each(function(_, obj) {
        var path = $(obj).text().trim();
        var title = ($(obj).attr('data-title') || '').trim();
        if (!path) {
            return;
        }
        if (path.charAt(0) === '/') {
            path = path.substring(1);
        }
        if (path.charAt(path.length - 1) === '/') {
            path = path.substring(0, path.length - 1);
        }
        if (!path.length) {
            return;
        }

        var strArray = path.split('/');
        for (var level = 1; level < strArray.length; level++) {
            var key = level + strArray[level];
            if (level === strArray.length - 1) {
                if (title) {
                    directoryTitles[path] = title;
                }
                tree.add(level + path, (level - 1) + strArray[level - 1]);
            } else {
                if (collection.indexOf(key) === -1) {
                    tree.add(key, (level - 1) + strArray[level - 1]);
                    collection.push(key);
                }
            }
        }
    });

    return tree;
}

//below is the construction of tree, get from github
function Node(data) {
    this.data = data;
    this.children = [];
}

function Tree() {
    this.root = null;
}

Tree.prototype.add = function(data, toNodeData) {
    var node = new Node(data);
    var parent = toNodeData ? this.findBFS(toNodeData) : null;
    if (parent) {
        parent.children.push(node);
    } else {
        if (!this.root) {
            this.root = node;
        } else {
            return 'Root node is already assigned';
        }
    }
};
Tree.prototype.remove = function(data) {
    if (!this.root) {
        return 'No root node found';
    }
    if (this.root.data === data) {
        this.root = null;
        return;
    }

    var queue = [this.root];
    while (queue.length) {
        var node = queue.shift();
        for (var i = 0; i < node.children.length; i++) {
            if (node.children[i].data === data) {
                node.children.splice(i, 1);
            } else {
                queue.push(node.children[i]);
            }
        }
    }
};
Tree.prototype.contains = function(data) {
    return this.findBFS(data) ? true : false;
};
Tree.prototype.findBFS = function(data) {
    if (!this.root) {
        return null;
    }
    var queue = [this.root];
    while (queue.length) {
        var node = queue.shift();
        if (node.data === data) {
            return node;
        }
        for (var i = 0; i < node.children.length; i++) {
            queue.push(node.children[i]);
        }
    }
    return null;
};
Tree.prototype._preOrder = function(node, fn) {
    if (node) {
        if (fn) {
            fn(node);
        }
        for (var i = 0; i < node.children.length; i++) {
            this._preOrder(node.children[i], fn);
        }
    }
};
Tree.prototype._postOrder = function(node, fn) {
    if (node) {
        for (var i = 0; i < node.children.length; i++) {
            this._postOrder(node.children[i], fn);
        }
        if (fn) {
            fn(node);
        }
    }
};
Tree.prototype.traverseDFS = function(fn, method) {
    var current = this.root;
    if (method) {
        this['_' + method](current, fn);
    } else {
        this._preOrder(current, fn);
    }
};
Tree.prototype.traverseBFS = function(fn) {
    var queue = [this.root];
    while (queue.length) {
        var node = queue.shift();
        if (fn) {
            fn(node);
        }
        for (var i = 0; i < node.children.length; i++) {
            queue.push(node.children[i]);
        }
    }
};
Tree.prototype.print = function() {
    if (!this.root) {
        return console.log('No root node found');
    }
    var newline = new Node('|');
    var queue = [this.root, newline];
    var string = '';
    while (queue.length) {
        var node = queue.shift();
        string += node.data.toString() + ' ';
        if (node === newline && queue.length) {
            queue.push(newline);
        }
        for (var i = 0; i < node.children.length; i++) {
            queue.push(node.children[i]);
        }
    }
    console.log(string.slice(0, -2).trim());
};
Tree.prototype.printByLevel = function() {
    if (!this.root) {
        return console.log('No root node found');
    }
    var newline = new Node('\n');
    var queue = [this.root, newline];
    var string = '';
    while (queue.length) {
        var node = queue.shift();
        string += node.data.toString() + (node.data !== '\n' ? ' ' : '');
        if (node === newline && queue.length) {
            queue.push(newline);
        }
        for (var i = 0; i < node.children.length; i++) {
            queue.push(node.children[i]);
        }
    }
    console.log(string.trim());
};

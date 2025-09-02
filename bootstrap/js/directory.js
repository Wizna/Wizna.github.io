$(document).ready(function() {
    $('#tipue_search_input').tipuesearch();
    
    var windowLoc = $(location).attr('pathname');
    if (windowLoc != '/directory/') {
        return;
    }
    console.log(windowLoc);
    

    var out = '<div class=\"span3\"> <div class=\"well\"> <div><ul class=\"nav nav-list\">';
    var tree = getDirectoryStructure();

    out += recurseTree(tree.root);

    out += "</ul></div></div></div>";

    $("#loadfiles").html(out);

    $('label.tree-toggler').click(function() {
        $(this).parent().children('ul.tree').toggle(200);
    });

    showPage();
});

function showPage() {
    document.getElementById("loader").style.display = "none";
    document.getElementById("placeHolder").style.display = "none";
    document.getElementById("loadfiles").style.display = "block";
}

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
    if (this.root.data === data) {
        this.root = null;
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

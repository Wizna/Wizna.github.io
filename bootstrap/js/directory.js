var directoryTitles = Object.create(null);

function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

$(document).ready(function() {
    if (typeof $.fn.tipuesearch === 'function') {
        $('#tipue_search_input').tipuesearch({
            'minimumLength': 1,    // Allow single-character queries for CJK languages
            'showTime': false,
            'showURL': false,
            'showTitleCount': false
        });
    }

    if (!$('#directory-tree').length) {
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

    if (node.isPost) {
        var path = node.data;
        var label = formatPostTitle(path);
        var url = node.url;

        return {
            html: '<li class="directory-item directory-post-item"><a href="' + escapeHtml(url) + '" class="directory-link">' + escapeHtml(label) + '</a></li>',
            count: 1
        };
    }

    var listResult = buildList(node.children, false);
    if (!listResult.count) {
        return { html: '', count: 0 };
    }

    var collapseId = 'collapse-' + sanitizeId(node.data);
    var displayName = formatCategoryTitle(node);

    // Bootstrap animates this wrapper's height; keep spacing inside it.
    var html = ''
        + '<li class="directory-item directory-category">'
        + '<button class="category-toggle" type="button" data-toggle="collapse" data-target="#' + collapseId + '" aria-controls="' + collapseId + '" aria-expanded="true">'
        + '<span class="category-name">' + escapeHtml(displayName) + '</span>'
        + '<span class="category-count">' + listResult.count + '</span>'
        + '</button>'
        + '<div class="collapse in directory-collapse" id="' + collapseId + '">'
        + '<div class="directory-sublist">'
        + listResult.html
        + '</div>'
        + '</div>'
        + '</li>';

    return { html: html, count: listResult.count };
}

function formatCategoryTitle(node) {
    return decodeURIComponent(node.data.split('/').pop()).replace(/-/g, ' ');
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

    if (node.isPost) {
        var path = node.data;
        var segments = path.split('/');
        var slug = decodeURIComponent(segments[segments.length - 1] || '');
        if (directoryTitles[path]) {
            return directoryTitles[path].toLowerCase();
        }
        slug = slug.replace(/^\d{4}-\d{2}-\d{2}-/, '');
        return slug.toLowerCase();
    }

    return formatCategoryTitle(node).toLowerCase();
}

function sanitizeId(value) {
    // Fixed-width UTF-16 encoding keeps case, punctuation and full paths unique.
    return value.split('').map(function(character) {
        return ('0000' + character.charCodeAt(0).toString(16)).slice(-4);
    }).join('');
}

function getDirectoryStructure() {
    var collection = Object.create(null);
    var tree = new Tree();
    tree.add('posts');
    directoryTitles = Object.create(null);

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
            var key = strArray.slice(0, level + 1).join('/');
            var parentKey = strArray.slice(0, level).join('/');
            if (level === strArray.length - 1) {
                if (title) {
                    directoryTitles[path] = title;
                }
                var postNode = tree.add(key, parentKey, true);
                postNode.url = $(obj).attr('data-url') || '/' + path + '/';
            } else {
                if (!collection[key]) {
                    tree.add(key, parentKey);
                    collection[key] = true;
                }
            }
        }
    });

    return tree;
}

//below is the construction of tree, get from github
function Node(data, isPost) {
    this.data = data;
    this.children = [];
    this.isPost = Boolean(isPost);
}

function Tree() {
    this.root = null;
}

Tree.prototype.add = function(data, toNodeData, isPost) {
    var node = new Node(data, isPost);
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
    return node;
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

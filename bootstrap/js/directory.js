function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

$(document).ready(function() {
    if (typeof $.fn.tipuesearch === 'function') {
        $('#tipue_search_input').tipuesearch({
            'minimumLength': 1    // Allow single-character queries for CJK languages
        });
    }

    var windowLoc = $(location).attr('pathname') || '';
    if (!/\/directory\/(index\.html)?$/.test(windowLoc)) {
        return;
    }
    var tree = getDirectoryStructure();
    var renderResult = buildList(tree.children, true);
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
    var loader = document.getElementById("loader");
    var placeHolder = document.getElementById("placeHolder");
    var loadfiles = document.getElementById("loadfiles");

    if (loader) loader.style.display = "none";
    if (placeHolder) placeHolder.style.display = "none";
    if (loadfiles) loadfiles.style.display = "block";
}

function buildList(children, isRoot) {
    if (!children || !Object.keys(children).length) {
        return { html: '', count: 0 };
    }

    var html = '<ul class="directory-list' + (isRoot ? ' directory-root' : '') + '">';
    var total = 0;
    var sorted = Object.keys(children).map(function(key) {
        return children[key];
    }).sort(function(a, b) {
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

    if (node.url) {
        return {
            html: '<li class="directory-item directory-post-item"><a href="' + node.url + '" class="directory-link">' + escapeHtml(node.title) + '</a></li>',
            count: 1
        };
    }

    var listResult = buildList(node.children, false);
    if (!listResult.count) {
        return { html: '', count: 0 };
    }

    var collapseId = 'collapse-' + sanitizeId(node.path);

    var html = ''
        + '<li class="directory-item directory-category">'
        + '<button class="category-toggle" type="button" data-toggle="collapse" data-target="#' + collapseId + '" aria-expanded="true">'
        + '<span class="category-name">' + escapeHtml(formatCategoryTitle(node.name)) + '</span>'
        + '<span class="category-count">' + listResult.count + '</span>'
        + '</button>'
        + '<div class="collapse in directory-sublist" id="' + collapseId + '">'
        + listResult.html
        + '</div>'
        + '</li>';

    return { html: html, count: listResult.count };
}

function createDirectoryNode(name, path) {
    return {
        name: name,
        path: path,
        children: {}
    };
}

function formatCategoryTitle(name) {
    return decodeURIComponent(name).replace(/-/g, ' ');
}

function formatPostTitle(path) {
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

    if (node.url) {
        return node.title.toLowerCase();
    }

    return decodeURIComponent(node.name).toLowerCase();
}

function sanitizeId(value) {
    var hash = 0;
    for (var i = 0; i < value.length; i++) {
        hash = ((hash << 5) - hash) + value.charCodeAt(i);
        hash = hash & hash;
    }

    return value.replace(/[^a-zA-Z0-9]+/g, '-').toLowerCase() + '-' + Math.abs(hash);
}

function getDirectoryStructure() {
    var root = createDirectoryNode('posts', 'posts');

    $(".post-direct").each(function(_, obj) {
        var path = ($(obj).data('path') || $(obj).text()).trim();
        var url = ($(obj).data('url') || ('../' + path)).trim();
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

        var segments = path.split('/');
        var current = root;

        for (var level = 1; level < segments.length; level++) {
            var segment = segments[level];
            var nodePath = segments.slice(0, level + 1).join('/');

            if (level === segments.length - 1) {
                current.children[nodePath] = {
                    title: title || formatPostTitle(path),
                    url: url
                };
                continue;
            }

            if (!current.children[nodePath]) {
                current.children[nodePath] = createDirectoryNode(segment, nodePath);
            }
            current = current.children[nodePath];
        }
    });

    return root;
}

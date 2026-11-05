const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const { JSDOM } = require('jsdom');

const root = path.resolve(__dirname, '..');
const source = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const post = (title, text = '', tags = '') => ({
    title, text, tags, url: '/posts/' + encodeURIComponent(title) + '/'
});

async function page(t, { posts, rows = [], options = {}, query } = {}) {
    const dom = new JSDOM(`<!doctype html><html><head><title>Blog</title></head><body>
        <input id="tipue_search_input">
        <div id="searchModal" class="modal"><div id="tipue_search_content"></div></div>
        ${posts ? '' : '<div id="loader"></div><div id="placeHolder"></div><div id="loadfiles"><div id="post-data"></div><div id="directory-count"></div><div id="directory-tree"></div></div>'}
        </body></html>`, {
        url: 'https://example.test/directory/' + (query ? '?q=' + encodeURIComponent(query) : ''),
        runScripts: 'dangerously'
    });
    t.after(() => dom.window.close());
    const win = dom.window;
    win.eval(source('bootstrap/js/jquery.min.js'));
    win.eval(source('bootstrap/js/bootstrap.min.js'));
    win.jQuery.support.transition = false;
    if (posts) {
        win.tipuesearch = { pages: posts };
        win.eval(source('assets/tipuesearch/tipuesearch_set.js'));
        win.eval(source('assets/tipuesearch/tipuesearch.min.js'));
        win.jQuery('#tipue_search_input').tipuesearch({
            minimumLength: 1, showTime: false, ...options
        });
    } else {
        for (const row of rows) {
            const span = win.document.createElement('span');
            span.className = 'post-direct';
            span.textContent = row.path;
            span.dataset.title = row.title;
            if (row.url) span.dataset.url = row.url;
            win.document.querySelector('#post-data').append(span);
        }
        win.eval(source('bootstrap/js/directory.js'));
        await new Promise((resolve) => win.jQuery(resolve));
    }
    return win;
}

function search(win, query) {
    const $ = win.jQuery;
    $('#tipue_search_input').val(query).trigger($.Event('keyup', { keyCode: 13 }));
    assert.equal(win.document.querySelector('.tipue_search_spinner'), null);
    return [...win.document.querySelectorAll('.tipue_search_content_title')].map((el) => el.textContent);
}

test('a URL search renders a correction payload as text', async (t) => {
    const query = 'javscript -<svg/onload=window.__tipuereviewprobe=1>';
    const win = await page(t, { posts: [post('JavaScript', 'javascript jquery js')], query });
    await new Promise((resolve) => win.setTimeout(resolve, 0));
    assert.ok(win.document.querySelector('#tipue_search_warning').textContent.includes(query));
    assert.equal(win.document.querySelector('#tipue_search_content svg'), null);
    assert.equal(win.__tipuereviewprobe, undefined);
    assert.equal(win.document.querySelectorAll('.tipue_search_content_title').length, 1);
});

test('titles and excerpts remain text while highlighting matches', async (t) => {
    const win = await page(t, {
        posts: [post('<img src=x onerror=window.__probe=1>', 'literal <svg/onload=window.__probe=1> and python')]
    });
    assert.equal(search(win, 'python').length, 1);
    assert.equal(win.document.querySelector('#tipue_search_content img, #tipue_search_content svg'), null);
    assert.ok(win.document.querySelector('.tipue_search_content_text').textContent.includes('<svg/'));
    assert.equal(win.document.querySelector('.tipue_search_content_bold').textContent, 'python');
});

test('regex punctuation is literal in normal and quoted queries', async (t) => {
    const win = await page(t, { posts: [post('Literal symbols', 'C++ C# .NET foo[0] [ ( \\ * + ? a.b')] });
    for (const query of ['[', '(', '\\', '*', '+', '?', 'a.b', '"["', '"("', '"C++"']) {
        assert.equal(search(win, query).length, 1, query);
    }
    assert.equal(search(win, '".*"').length, 0);
});

test('quoted phrases do not interpret regex wildcards', async (t) => {
    const win = await page(t, { posts: [post('Literal', 'a.b'), post('Other', 'axb')] });
    assert.deepEqual(search(win, '"a.b"'), ['Literal']);
});

test('symbol keywords, English word boundaries, and Chinese substrings work', async (t) => {
    const win = await page(t, {
        posts: [post('Symbols', 'C++ C# .NET python'), post('中文算法笔记'), post('Other', 'pythonic')]
    });
    for (const query of ['C++', 'C#', '.NET', 'python']) {
        assert.deepEqual(search(win, query), ['Symbols'], query);
    }
    assert.deepEqual(search(win, '算法'), ['中文算法笔记']);
});

test('exclusions are independent of keyword order and field', async (t) => {
    const win = await page(t, {
        posts: [post('Keep', 'python ranking'), post('Docker', 'python ranking'), post('Tag exclusion', 'python ranking', 'docker')]
    });
    assert.deepEqual(search(win, 'python -docker'), ['Keep']);
    assert.deepEqual(search(win, '-docker python'), ['Keep']);
    assert.deepEqual(search(win, 'python -docker ranking'), ['Keep']);
    assert.deepEqual(search(win, '-docker'), []);
});

test('all positive keywords must match', async (t) => {
    const win = await page(t, { posts: [post('Both', 'python ranking'), post('Partial', 'python')] });
    assert.deepEqual(search(win, 'python ranking'), ['Both']);
    assert.deepEqual(search(win, 'python\tranking'), ['Both']);
});

test('pagination preserves the choice to search without correction', async (t) => {
    const win = await page(t, {
        posts: [post('First', 'misspelled'), post('Second', 'misspelled'), post('Correction', 'corrected')],
        options: { show: 1 }
    });
    win.tipuesearch_replace = { words: [{ word: 'misspelled', replace_with: 'corrected' }] };
    assert.deepEqual(search(win, 'misspelled'), ['Correction']);
    win.jQuery('#tipue_search_replaced').trigger('click');
    assert.equal(win.document.querySelector('.tipue_search_content_title').textContent, 'First');
    win.jQuery('#tipue_search_foot_boxes a').last().trigger('click');
    assert.equal(win.document.querySelector('.tipue_search_content_title').textContent, 'Second');
    assert.equal(win.document.querySelector('#tipue_search_warning'), null);
});

test('same-named subcategories remain in their own branches', async (t) => {
    const win = await page(t, {
        rows: [{ path: '/posts/Coding/Python/one/', title: 'One' }, { path: '/posts/Learning/Python/two/', title: 'Two' }]
    });
    const roots = [...win.document.querySelector('.directory-root').children];
    assert.equal(win.document.querySelector('#directory-count').textContent, '2');
    assert.equal(roots.length, 2);
    assert.deepEqual(roots.map((node) => [...node.querySelectorAll('.directory-link')].map((link) => link.textContent)), [['One'], ['Two']]);
    assert.equal(win.document.querySelectorAll('.directory-link').length, 2);
});

test('case and punctuation produce unique, independently working collapse IDs', async (t) => {
    const win = await page(t, {
        rows: ['A-B', 'A_B', 'a-b', '中文'].map((category, i) => ({ path: '/posts/Coding/' + category + '/post/', title: 'Post ' + i }))
    });
    const buttons = [...win.document.querySelectorAll('.category-toggle')];
    const ids = buttons.map((button) => button.getAttribute('aria-controls'));
    assert.equal(new Set(ids).size, ids.length);
    const button = buttons.find((el) => el.querySelector('.category-name').textContent === 'A_B');
    win.jQuery(button).trigger('click');
    assert.equal(button.getAttribute('aria-expanded'), 'false');
    for (const other of buttons.filter((el) => el !== button)) {
        assert.equal(win.document.getElementById(other.getAttribute('aria-controls')).classList.contains('in'), true);
    }
});

test('directory links retain their canonical URL, base path, and title', async (t) => {
    const url = '/preview/posts/Coding/Python/one/';
    const win = await page(t, { rows: [{ path: '/posts/Coding/Python/one/', title: '<b>One "two"</b>', url }] });
    const link = win.document.querySelector('.directory-link');
    assert.equal(link.getAttribute('href'), url);
    assert.equal(link.textContent, '<b>One "two"</b>');
    assert.equal(link.querySelector('b'), null);
});

test('deployed articles generate TOCs with valid destinations', {
    skip: !process.env.SITE_URL && 'set SITE_URL after GitHub Pages deploys'
}, async () => {
    const files = [
        'posts/Learning/Review/2019-06-03-Machine learning/index.html',
        'posts/Learning/Review/2019-06-01-Algorithms/index.html',
        'posts/Coding/Python/2020-07-24-Python cheat sheet 2/index.html',
        'posts/Learning/Reading/2019-11-01-所读书籍的评分/index.html'
    ];
    for (const file of files) {
        const url = new URL(file, process.env.SITE_URL.replace(/\/?$/, '/'));
        const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
        assert.equal(response.ok, true, url.href);
        const dom = new JSDOM(await response.text());
        try {
            const doc = dom.window.document;
            const links = [...doc.querySelectorAll('#markdown-toc a')];
            assert.ok(links.length > 0, file);
            for (const link of links) assert.ok(doc.getElementById(decodeURIComponent(link.hash.slice(1))), link.hash);
            assert.equal([...doc.querySelectorAll('.entry li')].some((el) => /^TOC(?:\s|$)/.test(el.textContent.trim())), false);
        } finally {
            dom.window.close();
        }
    }
});

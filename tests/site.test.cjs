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

async function page(t, { posts, rows = [], options = {}, query, withForm = false, withUi = false } = {}) {
    const dom = new JSDOM(`<!doctype html><html><head><title>Blog</title></head><body>
        ${withForm ? '<form role="search">' : ''}<input id="tipue_search_input">${withForm ? '<button type="submit">Search</button></form>' : ''}
        <div id="searchModal" class="modal" tabindex="-1"><div class="modal-dialog"><div class="modal-content">
            <p class="search-modal__subtitle"></p><button class="search-modal__close" data-dismiss="modal">Close</button>
            <div class="search-modal__body"><div id="tipue_search_content"></div></div>
        </div></div></div>
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
    if (withUi) {
        win.eval(source('assets/js/site-ui.js'));
        await new Promise((resolve) => win.jQuery(resolve));
        win.jQuery.support.transition = false;
    }
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

test('search submits through its form and keeps each result together', async (t) => {
    const win = await page(t, { posts: [post('Ranking', 'ranking notes')], withForm: true });
    const form = win.document.querySelector('form');
    win.document.querySelector('#tipue_search_input').value = 'ranking';
    const submitted = new win.Event('submit', { bubbles: true, cancelable: true });
    form.dispatchEvent(submitted);
    assert.equal(submitted.defaultPrevented, true);
    const results = win.document.querySelectorAll('.tipue-search-result');
    assert.equal(results.length, 1);
    assert.equal(results[0].querySelector('.tipue_search_content_title').textContent, 'Ranking');
    assert.ok(results[0].querySelector('.tipue_search_content_text').textContent.includes('notes'));
});

test('result pagination is keyboard reachable and returns to the first result', async (t) => {
    const win = await page(t, { posts: [post('First', 'ranking'), post('Second', 'ranking')], options: { show: 1 } });
    search(win, 'ranking');
    const body = win.document.querySelector('.search-modal__body');
    body.scrollTop = 500;
    const next = win.document.querySelector('#tipue_search_foot_boxes a');
    assert.equal(next.getAttribute('href'), '#tipue_search_content');
    win.jQuery(next).trigger('click');
    assert.equal(win.document.querySelector('.tipue_search_content_title').textContent, 'Second');
    assert.equal(body.scrollTop, 0);
    assert.equal(win.document.activeElement, win.document.querySelector('.tipue_search_content_title a'));
    assert.equal(win.document.querySelector('#tipue_search_foot_boxes [aria-current="page"]').textContent, '2');
    assert.equal(win.location.hash, '');
});

test('search dialog closes back to its opener and renders query text safely', async (t) => {
    const win = await page(t, { posts: [post('Notes', 'ranking')], withForm: true, withUi: true });
    const input = win.document.querySelector('#tipue_search_input');
    input.value = '<svg onload=alert(1)>';
    input.focus();
    win.jQuery(input.form).trigger('submit');
    assert.equal(win.document.activeElement, win.document.querySelector('.search-modal__close'));
    assert.ok(win.document.querySelector('.search-modal__subtitle').textContent.includes(input.value));
    assert.equal(win.document.querySelector('.search-modal__subtitle svg'), null);
    win.jQuery('.search-modal__close').trigger('click');
    assert.equal(win.document.activeElement, input);
    assert.equal(win.document.body.classList.contains('modal-open'), false);
});

test('theme defaults to dark, toggles to light, and persists the choice', async (t) => {
    const dom = new JSDOM('<!doctype html><html><body><button data-theme-toggle hidden></button></body></html>', {
        url: 'https://example.test/',
        runScripts: 'dangerously'
    });
    t.after(() => dom.window.close());
    const win = dom.window;
    win.eval(source('assets/js/theme.js'));
    win.document.dispatchEvent(new win.Event('DOMContentLoaded'));
    const button = win.document.querySelector('[data-theme-toggle]');
    assert.equal(win.document.documentElement.getAttribute('data-theme'), 'dark');
    assert.equal(button.hidden, false);
    assert.equal(button.getAttribute('aria-pressed'), 'false');
    button.click();
    assert.equal(win.document.documentElement.getAttribute('data-theme'), 'light');
    assert.equal(button.getAttribute('aria-pressed'), 'true');
    assert.equal(win.localStorage.getItem('site-theme'), 'light');
});

function themePage(t, setup) {
    const dom = new JSDOM('<!doctype html><html><body><button data-theme-toggle hidden></button></body></html>', {
        url: 'https://example.test/',
        runScripts: 'dangerously'
    });
    t.after(() => dom.window.close());
    const win = dom.window;
    if (setup) setup(win);
    win.eval(source('assets/js/theme.js'));
    win.document.dispatchEvent(new win.Event('DOMContentLoaded'));
    return win;
}

test('theme switches hold CSS transitions off for two frames', async (t) => {
    const win = themePage(t);
    const root = win.document.documentElement;
    win.document.querySelector('[data-theme-toggle]').click();
    assert.equal(root.getAttribute('data-theme'), 'light');
    assert.equal(root.classList.contains('theme-switching'), true);
    await new Promise((resolve) => win.setTimeout(resolve, 80));
    assert.equal(root.classList.contains('theme-switching'), false);
});

test('theme switches cross-fade through View Transitions and persist at once', async (t) => {
    let update;
    let finish;
    const win = themePage(t, (w) => {
        w.document.startViewTransition = (callback) => {
            update = callback;
            return { finished: new Promise((resolve) => { finish = resolve; }) };
        };
    });
    const root = win.document.documentElement;
    win.document.querySelector('[data-theme-toggle]').click();
    assert.equal(root.getAttribute('data-theme'), 'dark');
    assert.equal(root.classList.contains('theme-switching'), true);
    assert.equal(win.localStorage.getItem('site-theme'), 'light');
    update();
    assert.equal(root.getAttribute('data-theme'), 'light');
    assert.equal(root.classList.contains('theme-switching'), true);
    finish();
    await new Promise((resolve) => win.setTimeout(resolve, 0));
    assert.equal(root.classList.contains('theme-switching'), false);
});

test('theme switches skip the cross-fade when reduced motion is requested', async (t) => {
    let started = 0;
    const win = themePage(t, (w) => {
        w.matchMedia = () => ({ matches: true });
        w.document.startViewTransition = () => {
            started += 1;
            return { finished: Promise.resolve() };
        };
    });
    win.document.querySelector('[data-theme-toggle]').click();
    assert.equal(started, 0);
    assert.equal(win.document.documentElement.getAttribute('data-theme'), 'light');
});

test('reduced motion removes Bootstrap transition waits and restores them afterwards', async (t) => {
    const dom = new JSDOM(`<!doctype html><html><body>
        <input id="tipue_search_input">
        <div id="searchModal" class="modal fade" tabindex="-1"></div>
        <button data-toggle="collapse" data-target="#panel">Toggle</button>
        <div id="panel" class="collapse in"></div>
    </body></html>`, { url: 'https://example.test/', runScripts: 'dangerously' });
    t.after(() => dom.window.close());
    const win = dom.window;
    const ready = () => new Promise((resolve) => win.jQuery(resolve));
    win.eval(source('bootstrap/js/jquery.min.js'));
    win.eval(source('bootstrap/js/bootstrap.min.js'));
    await ready();
    // jsdom has no CSS transitions, so pretend the browser supports them.
    const capable = { end: 'transitionend' };
    win.jQuery.support.transition = capable;
    const listeners = [];
    const media = { matches: true, addEventListener: (type, listener) => listeners.push(listener) };
    win.matchMedia = () => media;
    win.eval(source('assets/js/site-ui.js'));
    await ready();

    const $ = win.jQuery;
    const panel = win.document.getElementById('panel');
    const toggle = win.document.querySelector('[data-toggle="collapse"]');
    assert.equal($.support.transition, false);
    assert.equal($('#searchModal').hasClass('fade'), false);
    toggle.click();
    assert.equal(panel.className, 'collapse');

    media.matches = false;
    listeners.forEach((listener) => listener());
    assert.equal($.support.transition, capable);
    assert.equal($('#searchModal').hasClass('fade'), true);
    toggle.click();
    assert.equal(panel.classList.contains('collapsing'), true);
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

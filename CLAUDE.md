# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jekyll personal blog on GitHub Pages, with posts in English and Chinese. Pushing to `master` triggers the GitHub Pages build and deploy; there is no CI config in the repo. The site is dark by default with a light-mode toggle.

## Commands

Tests are the verification that works with only Node installed (Node >= 18, `jsdom` pinned in `package.json`):

```bash
npm ci
npm test                                                       # node --test tests/site.test.cjs
node --test --test-name-pattern="theme" tests/site.test.cjs    # run a single test by title
SITE_URL=https://wizna.github.io/ npm test                     # also runs the deployed-TOC test (skipped otherwise)
```

Jekyll:

```bash
bundle exec jekyll serve --livereload   # add --drafts to preview drafts
bundle exec jekyll build
bundle exec jekyll doctor               # permalink/config problems
```

`Gemfile` and `Gemfile.lock` are gitignored and absent from this checkout. Plugins (`jekyll-paginate`, `jekyll-sitemap`, `jekyll-feed`) are declared under `plugins:` in `_config.yml`.

A local build works on the macOS system Ruby (2.6.10), but the newest gems need Ruby 3+, so pin the versions that still support 2.6. This combination built the whole site (47 files) and matched the live HTML apart from the removed image script (checked on one post):

```bash
# Pre-pin the dependencies that dropped Ruby 2.6 support, one gem per call (gem rejects -v with several gems).
# Assembled from the steps that worked here; not re-run as a single script.
for g in ffi:1.15.5 i18n:1.14.8 public_suffix:4.0.7 jekyll:3.10.0 jekyll-paginate:1.1.0 kramdown-parser-gfm:1.1.0; do
  gem install --user-install --no-document "$g"
done
export PATH="$HOME/.gem/ruby/2.6.0/bin:$PATH"                     # user-installed binaries are not on PATH by default
jekyll build --config _config.yml,/tmp/local.yml --destination /tmp/site-out
```

`/tmp/local.yml` is a throwaway override. Its `plugins:` list replaces the one in `_config.yml`, so list only `jekyll-paginate` (`jekyll-sitemap` and `jekyll-feed` are slow to install and irrelevant to UI checks). Add the two PDFs and `assets/summary` to `exclude:`. Add a `defaults:` entry that gives `type: posts` the layout `post`. GitHub Pages does this implicitly with its `jekyll-default-layout` plugin; a plain local build does not, so posts come out without the site chrome. Serve the output with `python3 -I -m http.server --bind 127.0.0.1 --directory /tmp/site-out`. Expect harmless "Liquid syntax error" warnings for `{{` inside a few post bodies (math and Jinja examples), and a `+08:00` instead of `+00:00` offset in post dates because Pages builds in UTC.

## Architecture

### Layouts and the `.entry` contract

`default.html` is the root template. `post.html` and `page.html` extend it and wrap content in `<article>` + `<div class="entry">`; `index.html` uses `default` directly and renders the paginated list (it must stay `.html` for `jekyll-paginate`).

Several things depend on the `.entry` class, so any new layout must use it:
- `assets/js/site-ui.js`, which wraps `.entry table` in a scrollable `.table-scroll` div
- the deployed-TOC test, which looks inside `.entry`

Image hints are added at build time, not in the browser. `default.html` renders `{{ content }}` through `_includes/optimize-images.html`, which adds `decoding="async"` and `loading="lazy"` to every `<img>` except the first one on the page, and appends size parameters to Pexels (`auto=compress&cs=tinysrgb&w=800`) and Unsplash (`w=800&q=80`) URLs that have none. It must stay in Liquid: the browser requests each `src` while parsing, so a script that rewrites `src` or `loading` afterwards cannot stop the first request and makes the browser fetch the image twice. Only images in `content` are processed; the header avatar sets its own attributes. The include only matches `<img ` followed by a space and a `src="..."` attribute, so other spellings are left untouched.

### Theming

- `assets/js/theme.js` loads synchronously in `<head>`, before the stylesheet. It sets `data-theme` on `<html>` (default `dark`; `localStorage['site-theme'] === 'light'` switches it), updates `meta[name=theme-color]`, and wires `[data-theme-toggle]`. That button is rendered `hidden` and revealed by the script.
- A toggle dispatches a `site:themechange` event. `assets/js/diagrams.js` listens to it to re-render Mermaid in the matching theme.
- While the theme changes, `theme.js` puts `.theme-switching` on `<html>`. `_sass/_site.scss` uses it to turn every `transition` off, because per-element colour transitions (links, nav, buttons) otherwise lag behind the instantly changing background. With `document.startViewTransition` the page cross-fades instead, unless reduced motion is requested. The saved choice is written immediately, but the DOM changes inside the transition callback. Any new colour transition is covered automatically; do not add `!important` overrides that outrank `.theme-switching`.
- All colors are CSS custom properties defined in `:root` (dark) and `:root[data-theme="light"]` in `_sass/_site.scss`. `bootstrap/css/directory.css` consumes the same variables. Add new colors as variables with a value in both themes rather than hard-coding them.

### Sass

`style.scss` contains only `@import`s (reset, variables, highlights, svg-icons, site). The real styles, including the search modal, are in `_sass/_site.scss`. `_sass/_variables.scss` (`$blue`, etc.) is legacy. `_sass/_svg-icons.scss` is ~54 KB of base64 SVG, so avoid reading it. `bootstrap/css/directory.css` is separate and linked from `directory.html`.

### Directory page (Liquid -> JS)

`directory.html` (permalink `/directory/`, layout `page`) emits hidden `<span class="post-direct" data-title data-url>` elements, one per post, with the post URL as text. `bootstrap/js/directory.js` builds a tree from the URL path segments and renders Bootstrap 3 collapsible panels.
- Tree position comes from the span text; the link `href` comes from `data-url`, which is `relative_url`-aware.
- Collapse IDs are hex-encoded UTF-16 of the full path, so same-named subcategories and names differing only by case or punctuation stay unique.
- Titles and URLs go through `escapeHtml`. Tests assert that markup in titles stays text.
- `directory.js` is loaded on every page and does two jobs. It initializes Tipue search with `minimumLength: 1` (so single CJK characters work), and it builds the tree only when `#directory-tree` exists.

### Search (Tipue, locally forked)

- `assets/tipuesearch/tipuesearch_content.js` has front matter, so Jekyll processes it as a Liquid template that emits `var tipuesearch = {pages: [...]}` from all of `site.posts` (title, text, tags, url).
- The Tipue files are a locally maintained fork. Per `README.md`, upstream is unreachable and the newest cdnjs build (5.0.0) is older. The fork adds Chinese substring matching, literal handling of regex punctuation, text-only rendering of titles and queries, and modal integration. Do not replace it with a CDN or upstream copy.
- Tipue loads on every page because `_layouts/default.html` itself sets `tipue_search_active: true` in its front matter. The per-page flag documented elsewhere is effectively redundant.
- The header search form opens the Bootstrap modal `#searchModal` and renders results into `#tipue_search_content`. `site-ui.js` handles focus return to the opener, sets the subtitle text, and applies `prefers-reduced-motion`: it drops the modal's `.fade` and sets `$.support.transition = false`. The second part matters because the reduced-motion rule in `_site.scss` removes every CSS transition, yet Bootstrap 3's collapse still waits for a `transitionend` (plus a 350 ms fallback timer) and ignores clicks while it waits. With transition support off, Bootstrap finishes immediately. The original value is restored if the preference changes back. Do not patch `bootstrap/`; fix this kind of problem from our own scripts.

### Tests

`tests/site.test.cjs` loads the real source files into jsdom with `win.eval` (jQuery, then Bootstrap, then `site-ui.js` / Tipue / `directory.js` / `theme.js`). Those files must stay classic scripts with no module syntax and no new dependencies beyond jQuery/Bootstrap. The `tests/` folder and `node_modules` are in `exclude`, so they are not published. The deployed-TOC test hard-codes four post URLs, so renaming or moving those posts breaks it.

### Third-party resources

| Resource | Where | Loading |
|----------|-------|---------|
| MathJax 2.7.9 (cdnjs) | `_includes/meta.html` | Every page; `$...$` inline only, skips `pre` |
| Mermaid 10.6.1 (jsdelivr) | `assets/js/diagrams.js` | Only when `.language-mermaid` / `.mermaid` nodes exist |
| Disqus | `_includes/disqus.html` | `post.html` only |
| Google Analytics | `_includes/analytics.html` | Every page |
| Bootstrap 3.3.7, jQuery 3.7.1 | `bootstrap/` (vendored) | Every page; do not use Bootstrap 4/5 APIs |

Local scripts and stylesheets are referenced with `?v={{ site.time | date: '%s' }}` for cache busting. Keep that when adding assets.

## Jekyll configuration gotchas

- **Dual permalinks**: the global `permalink: /:title/` applies to pages (`about.md` -> `/about/`). The `posts` collection overrides it with `/posts/:path/`, so the folder path under `_posts/` is part of the URL. Moving a post changes its URL.
- **`post.categories` is empty** because `posts` is declared explicitly under `collections:`. Derive categories from `post.path` (`index.html` already does this).
- Pagination is 5 posts per page at `/page:num/`.

## Content

- New post: `_posts/<Category>/<Subcategory>/YYYY-MM-DD-title.md`. Reuse the existing category folders (`Coding/*`, `Handbook`, `Learning/*`) and match their casing.
- Front matter is mostly omitted. The title is derived from the filename. Add front matter only to override (`title:`, `excerpt:`, `exclude_from_search: true`).
- Table of contents: put `* TOC` followed by `{:toc}` at the top of the post. The deployed test checks that `#markdown-toc` links resolve and that no literal "TOC" list item remains.
- Diagrams use a `mermaid` code fence. Math uses `$...$`.
- Do not read `assets/Simple-Java.pdf`, `assets/paper.pdf`, or the PDFs in `assets/summary/`.

## Style and commits

- Four-space indentation in HTML, Liquid, and JS.
- Recent commits use conventional prefixes (`feat:`, `fix:`); content uses `post: add <title>`. Keep the first line short and imperative.
- Include screenshots in PRs that change navigation, the directory tree, the search modal, or either theme.

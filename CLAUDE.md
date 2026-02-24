# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jekyll-based personal blog on GitHub Pages. Features programming tutorials, algorithm reviews, and technical notes in English and Chinese. Deploys automatically on push to `master`.

## Development Commands

```bash
# Local development with live reload
bundle exec jekyll serve --livereload

# Preview with drafts
bundle exec jekyll serve --drafts

# Build only (no server)
bundle exec jekyll build

# Diagnose config/permalink issues
bundle exec jekyll doctor
```

First-time setup requires `gem install bundler jekyll`. Note: `Gemfile` and `Gemfile.lock` are gitignored — dependencies are not version-locked. Jekyll plugins (`jekyll-paginate`, `jekyll-sitemap`, `jekyll-feed`) are declared via the `gems:` key in `_config.yml`.

## Layout Inheritance & Template Chain

```
default.html          ← Root template (head, nav, footer, image optimization JS, conditional Mermaid)
├── post.html         ← layout: default — wraps content in <article class="post"><div class="entry">
├── page.html         ← layout: default — wraps content in <article class="page"><div class="entry">
└── index.html        ← layout: default — homepage with paginator (5 posts/page)
```

**Critical selector: `.entry`** — Used by post.html, page.html, and index.html. The image optimization script in default.html (lines 110-147) targets `.post .entry`, `article.post`, and `.entry` containers. Any new layout must use this class for images to get lazy loading and URL optimization.

### Includes (`_includes/`)

- **meta.html** — `<head>` content: charset, viewport, OG/Twitter meta tags, and **MathJax 2.7.9 config + script** (loads on every page)
- **analytics.html** — Google Analytics via GTM (`G-LMJJHCRNEZ`), loaded at end of `<body>` when `site.google_analytics` is set
- **disqus.html** — Disqus comment embed (shortname: `ruiming-huangs-blog`), included only in `post.html`
- **svg-icons.html** — Social media footer links rendered as SVG icon sprites

### Root-Level Site Pages

| File | Permalink | Layout | Notes |
|------|-----------|--------|-------|
| `index.html` | `/` | default | Homepage with paginator (5 posts/page) |
| `directory.html` | `/directory/` | page | Post browser + search; has `tipue_search_active: true` and `exclude_from_search: true` |
| `about.md` | `/about/` | page | Profile, hobbies, contact info |
| `404.md` | (auto) | page | Custom 404 with back-to-home link |

## Content Guidelines

- **New post**: Create `YYYY-MM-DD-title.md` in `_posts/<Category>/<Subcategory>/`. The folder path becomes the URL under `/posts/...`. Existing category tree:
  ```
  _posts/
  ├── Coding/
  │   ├── Python/          # 18 posts — cheat sheets, data structures, libraries
  │   ├── JS-HTML-CSS/     # 2 posts
  │   └── Ai/              # 1 post
  ├── Handbook/            # 3 posts — tool shortcuts (PyCharm, IntelliJ, etc.)
  └── Learning/
      ├── RWTH/            # 1 post — course summaries
      ├── Reading/         # 1 post — book reviews
      └── Review/          # 5 posts + subcategories:
          ├── Advertising/ # 2 posts
          └── Recommendation/ # 2 posts
  ```
- **Front matter is mostly optional**: Most posts have none. Jekyll infers the title from the filename (strips date prefix, converts hyphens to spaces). Add front matter only to override defaults.
- **Available front matter flags**:
  - `layout: post` — default for posts; rarely needed explicitly
  - `title:` — override the filename-derived title
  - `excerpt:` — custom excerpt for homepage listing
  - `tipue_search_active: true` — enable search indexing on non-default pages
  - `exclude_from_search: true` — exclude from Tipue search index
- **Math**: Use `$...$` for inline math on any page (MathJax 2.7.9 loads globally via `_includes/meta.html`)
- **Mermaid diagrams**: Use `mermaid` as code fence language; loaded conditionally only when detected
- **Table of contents**: Add `* TOC {:toc}` at the top of post body for Kramdown auto-generated TOC
- **Code blocks**: Triple backticks with language identifier (Rouge highlighter)

## Custom Architecture Components

### Automatic Image Optimization

[_layouts/default.html](_layouts/default.html) (lines 110-147) runs client-side post-processing on images inside `.post .entry`, `article.post`, and `.entry` containers:

- **Pexels** (`images.pexels.com`): appends `?auto=compress&cs=tinysrgb&w=800`
- **Unsplash** (`images.unsplash.com`): appends `?w=800&q=80`
- **First image**: eager-loaded (LCP candidate); all others get `loading="lazy"`
- **All images**: get `decoding="async"`

Images from other hosts receive lazy loading and async decoding but no compression parameters.

### Directory System (Jekyll → JS Pipeline)

The [directory page](directory.html) uses a two-stage pipeline:

1. **Liquid stage** (`directory.html` lines 21-23): Iterates `site.posts` and emits hidden `<span class="post-direct" data-title="...">{{ post.url }}</span>` elements.
2. **JS stage** ([bootstrap/js/directory.js](bootstrap/js/directory.js)): `getDirectoryStructure()` parses these spans from the DOM, builds a hierarchical tree, then `buildList()`/`buildNode()` render it as Bootstrap collapsible panels.

Key behaviors: date prefixes (`YYYY-MM-DD-`) stripped from slugs, `data-title` preferred over slug-derived titles, posts sorted alphabetically within categories. The directory JS only executes when `pathname` matches `/directory/` — it's safe to load on all pages. Custom tree styling is in `bootstrap/css/directory.css`.

### Client-Side Search (Tipue Search)

- [assets/tipuesearch/tipuesearch_content.js](assets/tipuesearch/tipuesearch_content.js): Liquid template that builds a JSON search index from `site.posts` at build time, respecting `exclude_from_search` front matter
- **Bilingual support**: `tipuesearch_set.js` defines English + Chinese stop words and bilingual UI strings
- **Minimum query length**: 1 character, configured in [bootstrap/js/directory.js](bootstrap/js/directory.js) line 6 (`minimumLength: 1`), not in Tipue's own config — enables single-character CJK queries
- **Conditional loading**: Tipue JS/CSS only loads on pages with `tipue_search_active: true` front matter (set in `default.html` and `directory.html`)
- **Search modal**: Bootstrap 3 `.modal` triggered by search input, results rendered into `#tipue_search_content`, heavily styled via custom Sass in `style.scss` (lines 295-598)

### Conditional Resource Loading (Three Patterns)

| Resource | Strategy | Trigger |
|----------|----------|---------|
| Tipue Search | Front matter flag | `tipue_search_active: true` on page or layout |
| Mermaid 10.6.1 | DOM detection | `.language-mermaid` or `.mermaid` class found after DOMContentLoaded |
| MathJax 2.7.9 | Always loaded | Unconditional via `_includes/meta.html` on every page |

## Jekyll Configuration

Key settings in [_config.yml](_config.yml):

- Kramdown with GFM, Rouge syntax highlighter
- Plugins: `jekyll-paginate`, `jekyll-sitemap`, `jekyll-feed`
- **Dual permalink system**: default `permalink: /:title/` applies to non-collection pages (`about.md` → `/about/`); posts use the collection override `permalink: /posts/:path/` (folder structure becomes URL path)
- Pagination: 5 posts per page (`/page:num/`)
- Homepage is `index.html` (not `.md`) — required by `jekyll-paginate`

## Sass Architecture

- `style.scss` (607 lines) — main stylesheet, imports partials below
- `_sass/_reset.scss` — Meyer reset + box-sizing
- `_sass/_variables.scss` — colors (`$blue: #4183C4`), fonts (Helvetica/Georgia), mobile breakpoint (640px)
- `_sass/_highlights.scss` — Rouge syntax theme (Solarized Dark)
- `_sass/_svg-icons.scss` — **39KB+ of base64-encoded SVG data** for social icons; avoid reading this file
- `bootstrap/css/directory.css` — custom styling for the directory tree collapsible panels

Search modal styling lives in `style.scss` lines 295-598 (gradients, blur, responsive flex layout).

## Large Binary Files (Do Not Read)

- `assets/Simple-Java.pdf` (~2.8 GB) and `assets/paper.pdf` (~7.5 GB) — reference PDFs, never read or process these
- `assets/summary/` — 11 PDF study summaries (~44 MB total)
- `_sass/_svg-icons.scss` — 39KB+ of base64-encoded SVG data

## Coding Style

- Four-space indentation in HTML, Liquid, and JavaScript files
- Use existing Bootstrap 3 class patterns (`data-toggle`, `collapse`, `modal`, `container`)
- Lowercase directory names with hyphens in `_posts/` paths
- Post filenames: `YYYY-MM-DD-descriptive-title-with-hyphens.md`

## Pre-PR Verification

1. Run `bundle exec jekyll build` to ensure compilation
2. Run `bundle exec jekyll doctor` to catch permalink/config issues
3. Verify Tipue search indexes new content (check generated `_site/assets/tipuesearch/tipuesearch_content.js`)
4. Confirm referenced assets resolve (images, CSS, JS)
5. Include screenshots when changing UI elements (navigation, directory tree, search modal, styling)

## Commit Conventions

Use short imperative messages (e.g., `update`, `fix`). Pattern for content: `post: add pycharm shortcuts`. Add body when introducing notable layout/asset changes.

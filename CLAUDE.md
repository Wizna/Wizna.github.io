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

First-time setup requires Ruby tooling: `gem install bundler jekyll`.

## Project Structure

- `_posts/<Category>/<Subcategory>/`: Blog posts (folder path becomes part of `/posts/...` permalink)
- `_layouts/`: Templates (`default.html`, `post.html`, `page.html`)
- `_includes/`: `meta.html` (SEO meta + MathJax), `disqus.html`, `analytics.html`, `svg-icons.html`
- `_sass/` + `style.scss`: Sass sources with top-level overrides
- `bootstrap/`, `assets/`: Vendored front-end libs (Bootstrap 3.3.7, jQuery 3.7.1, Tipue Search, directory tree)
- `images/`: Site imagery

## Content Guidelines

- **New post**: Create `YYYY-MM-DD-title.md` in `_posts/<Category>/<Subcategory>/`. The folder path becomes the URL under `/posts/...`
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

Key behaviors: date prefixes (`YYYY-MM-DD-`) stripped from slugs, `data-title` preferred over slug-derived titles, posts sorted alphabetically within categories.

### Client-Side Search (Tipue Search)

- [assets/tipuesearch/tipuesearch_content.js](assets/tipuesearch/tipuesearch_content.js): Liquid template that builds a JSON search index from `site.posts` at build time, respecting `exclude_from_search` front matter
- **Bilingual support**: `tipuesearch_set.js` defines English + Chinese stop words and bilingual UI strings
- **Minimum query length**: 1 character, configured in [bootstrap/js/directory.js](bootstrap/js/directory.js) line 6 (`minimumLength: 1`), not in Tipue's own config — enables single-character CJK queries
- **Conditional loading**: Tipue JS/CSS only loads on pages with `tipue_search_active: true` front matter (set in `default.html` and `directory.html`)

### Performance Optimizations

- **Conditional Mermaid**: Mermaid 10.6.1 loaded from CDN only when `.language-mermaid` or `.mermaid` classes detected
- **Unconditional MathJax**: MathJax 2.7.9 loads on every page via `_includes/meta.html`
- **Image optimization**: Automatic compression for Pexels/Unsplash URLs, lazy loading for non-first images
- **Vendored dependencies**: Bootstrap, jQuery, Tipue Search served locally (only Mermaid and MathJax use CDN)
- **Avatar preload**: WebP avatar preloaded via `<link rel="preload">` for faster LCP

## Jekyll Configuration

Key settings in [_config.yml](_config.yml):

- Kramdown with GFM, Rouge syntax highlighter
- Plugins: `jekyll-paginate`, `jekyll-sitemap`, `jekyll-feed`
- Page permalink: `/:title/`
- Posts permalink: `/posts/:path/`
- Pagination: 5 posts per page (`/page:num/`)

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

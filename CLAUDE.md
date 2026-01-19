# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jekyll-based personal blog on GitHub Pages with Jekyll Now theme. Features programming tutorials, algorithm solutions, and technical notes in English and Chinese.

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

First-time setup requires Ruby tooling: `gem install bundler jekyll`. Deployment is automatic on push to `master` via GitHub Pages.

## Project Structure

- `_posts/<Category>/<Subcategory>/`: Blog posts (folder path becomes part of `/posts/...` permalink)
- `_layouts/`: Templates (`default.html`, `post.html`, `page.html`)
- `_includes/`: Shared HTML fragments
- `_sass/` + `style.scss`: Sass sources with top-level overrides
- `bootstrap/`, `assets/`: Vendored front-end libs (Bootstrap, Tipue Search, directory tree)
- `images/`: Site imagery

## Content Guidelines

- **New post**: Create `YYYY-MM-DD-title.md` in `_posts/<Category>/<Subcategory>/`
- **Front matter**: Requires `layout: post` at minimum; add `tipue_search_active: true` for search indexing
- **Code blocks**: Use triple backticks with language identifier (Rouge highlighter)
- **Mermaid diagrams**: Use `mermaid` as code fence language
- **Search exclusion**: Add `exclude_from_search: true` to front matter

## Custom Architecture Components

### Dynamic Directory System

[bootstrap/js/directory.js](bootstrap/js/directory.js) builds a hierarchical tree view of all posts:

- `getDirectoryStructure()`: Parses `.post-direct` elements from Jekyll's post collection
- `buildList()`/`buildNode()`: Recursively renders Bootstrap collapsible UI
- Path parsing strips date prefixes (`YYYY-MM-DD-`) from post slugs
- Category nodes use Bootstrap's `data-toggle="collapse"` for expand/collapse

### Client-Side Search (Tipue Search)

- [assets/tipuesearch/tipuesearch_content.js](assets/tipuesearch/tipuesearch_content.js): Jekyll Liquid template generates search index from `site.posts`
- Search modal in [_layouts/default.html](_layouts/default.html)
- Conditional loading: Only loads on pages with `tipue_search_active: true` front matter
- Minimum query: 1 character (supports CJK single-character queries)

### Performance Optimizations

- **Conditional Mermaid**: [_layouts/default.html](_layouts/default.html) only loads Mermaid.js from CDN when `.language-mermaid` or `.mermaid` classes are present
- **Vendored dependencies**: Bootstrap 3.3.7, jQuery 3.7.1, Tipue Search served locally
- **Image preloading**: WebP avatar preloaded for faster LCP

## Jekyll Configuration

Key settings in [_config.yml](_config.yml):

- Kramdown with GFM, Rouge syntax highlighter
- Plugins: `jekyll-paginate`, `jekyll-sitemap`, `jekyll-feed`
- Page permalink: `/:title/`
- Posts permalink: `/posts/:path/`
- Pagination: 5 posts per page (`/page:num/`)

## Pre-PR Verification

Run `bundle exec jekyll build` before PRs to ensure compilation. Use `jekyll doctor` to catch permalink/config issues. Verify Tipue search indexes new content and that referenced assets resolve.

## Commit Conventions

Use short imperative messages (e.g., `update`, `fix`). Pattern for content: `post: add pycharm shortcuts`. Add body when introducing notable layout/asset changes.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Jekyll-based personal blog on GitHub Pages with Jekyll Now theme. Features programming tutorials, algorithm solutions, and technical notes in English and Chinese.

## Development Commands

```bash
# Local development (requires Jekyll installed)
jekyll serve

# Preview with drafts
jekyll serve --drafts

# Build only (no server)
jekyll build
```

No Gemfile present—dependencies must be installed manually if running locally. Deployment is automatic on push to `master` via GitHub Pages.

## Content Guidelines

- **New post**: Create `YYYY-MM-DD-title.md` in `_posts/<Category>/<Subcategory>/`
- **Front matter**: Requires `layout: post` at minimum
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

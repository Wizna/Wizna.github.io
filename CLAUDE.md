# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Jekyll-based personal blog hosted on GitHub Pages. The site uses Jekyll Now theme with custom modifications for a personal tech blog featuring programming tutorials, algorithm solutions, and technical notes in both English and Chinese.

## Site Structure

- **`_posts/`**: Blog posts organized by category
  - `Coding/`: Programming tutorials (Python, JS/HTML/CSS, AI subcategories)
  - `Learning/`: Course summaries, book reviews, and study notes (Reading, Review subdirectories)
  - `Handbook/`: Quick reference guides and cheat sheets
- **`_layouts/`**: Jekyll page templates (default.html, post.html, page.html)
- **`_includes/`**: Reusable template components (analytics, disqus, meta tags, svg-icons)
- **`_sass/`**: SCSS stylesheets for custom styling
- **`assets/`**: Static assets including PDFs, search functionality (Tipue Search), and images
- **`bootstrap/`**: Bootstrap 3.3.7 CSS/JS framework and custom directory functionality
- **`images/`**: Site images and favicon

## Key Features

- **Search functionality**: Tipue Search integration for client-side full-text search
- **Multilingual content**: Posts in both English and Chinese
- **Mermaid diagrams**: Conditional async loading for diagram rendering
- **Pagination**: Blog post pagination (5 posts per page)
- **Comments**: Disqus integration (shortname: `ruiming-huangs-blog`)
- **Analytics**: Google Analytics tracking (UA-96728449-1)
- **Directory page**: Custom JavaScript-based directory/index view with collapsible tree structure
- **Bootstrap integration**: Responsive design with Bootstrap 3.3.7 and jQuery 3.7.1

## Development Workflow

This site uses GitHub Pages' automatic Jekyll processing with no local build system:
- **Deployment**: Automatic on push to `master` branch via GitHub Pages
- **Local development**: Run `jekyll serve` if Jekyll is installed locally (no Gemfile present, manual setup required)
- **Testing changes**: Edit files directly and push to trigger GitHub Pages build
- **Dependencies**: All frontend libraries vendored locally except Mermaid.js (loaded from CDN when needed)

## Jekyll Configuration

Key settings in [_config.yml](_config.yml):
- Uses Kramdown with GitHub Flavored Markdown (GFM)
- Rouge syntax highlighter with Pygments CSS class
- Jekyll plugins: `jekyll-paginate`, `jekyll-sitemap`, `jekyll-feed`
- Page permalink: `/:title/`
- Posts collection permalink: `/posts/:path/`
- Site URL: `https://wizna.github.io/`

## Custom Architecture Components

### Dynamic Directory System
The directory page ([/directory](https://wizna.github.io/directory)) dynamically generates a hierarchical view of all posts:
- **Core logic**: [bootstrap/js/directory.js](bootstrap/js/directory.js) implements custom Tree data structure
- **Tree construction**: `getDirectoryStructure()` (lines 50-73) parses `.post-direct` elements from Jekyll's post collection
- **Recursive rendering**: `recurseTree()` (lines 30-48) generates Bootstrap collapsible UI with proper nesting
- **Path parsing**: Extracts directory structure from post permalinks, stripping date prefixes (`YYYY-MM-DD-`)
- **Bootstrap collapse**: Each category node uses Bootstrap's `data-toggle="collapse"` for expand/collapse functionality

### Client-Side Search
Full-text search without server-side processing:
- **Tipue Search**: Indexes all post content via Jekyll Liquid templates
- **Content generation**: [assets/tipuesearch/tipuesearch_content.js](assets/tipuesearch/tipuesearch_content.js) dynamically generated from Jekyll `site.posts`
- **Modal interface**: Search results display in Bootstrap modal ([_layouts/default.html:75-85](_layouts/default.html#L75-L85))
- **Conditional loading**: Search assets only load on pages with `tipue_search_active: true` front matter
- **Minimum query length**: 3 characters required for search

### Performance Optimizations
- **Conditional Mermaid loading**: [_layouts/default.html:29-47](_layouts/default.html#L29-L47) checks for `.language-mermaid` or `.mermaid` classes before loading Mermaid.js from CDN
- **Image preloading**: WebP avatar preloaded for faster LCP ([_layouts/default.html:16](_layouts/default.html#L16))
- **Vendored dependencies**: All core JS/CSS libraries served locally (Bootstrap, jQuery, Tipue Search)
- **Async script loading**: Mermaid.js loaded asynchronously with `script.async = true`

### Template Architecture
- **Layout hierarchy**: `default.html` → `post.html` or `page.html`
- **Multi-language support**: Chinese/English content with consistent navigation structure
- **Custom permalinks**: Posts use `/posts/:path/` to preserve category structure in URLs
- **Pagination**: 5 posts per page with pagination navigation (`paginate_path: /page:num/`)

## Content Guidelines

- **Post format**: YAML front matter with `layout` specification (usually `post`)
- **Filename convention**: `YYYY-MM-DD-title.md` in appropriate `_posts/` subdirectory
- **Category structure**: Determined by subdirectory path (e.g., `_posts/Coding/Python/` → `/posts/Coding/Python/`)
- **Code highlighting**: Use triple backticks with language identifier (rendered via Rouge)
- **Mermaid diagrams**: Use `mermaid` code fence language identifier
- **Multilingual**: Posts can be in English or Chinese; navigation and UI elements support both
- **Search exclusion**: Add `exclude_from_search: true` to front matter to exclude from Tipue Search index

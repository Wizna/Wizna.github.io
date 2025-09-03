# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Jekyll-based personal blog hosted on GitHub Pages. The site uses Jekyll Now theme with custom modifications for a personal tech blog featuring programming tutorials, algorithm solutions, and technical notes in both English and Chinese.

## Site Structure

- **`_posts/`**: Blog posts organized by category
  - `Coding/`: Programming tutorials and solutions (Python, JS/HTML/CSS)
  - `Learning/`: Course summaries, book reviews, and study notes
  - `Handbook/`: Quick reference guides and cheat sheets
- **`_layouts/`**: Jekyll page templates (default.html, post.html, page.html)
- **`_includes/`**: Reusable template components (analytics, disqus, meta tags, svg-icons)
- **`_sass/`**: SCSS stylesheets for custom styling
- **`assets/`**: Static assets including PDFs, search functionality (Tipue Search), and images
- **`bootstrap/`**: Bootstrap CSS/JS framework and custom directory functionality
- **`images/`**: Site images and favicon

## Key Features

- **Search functionality**: Tipue Search integration for site-wide content search
- **Multilingual content**: Posts in both English and Chinese
- **Mermaid diagrams**: Support for diagram rendering in posts via mermaid.min.js
- **Pagination**: Blog post pagination (5 posts per page)
- **Comments**: Disqus integration for post comments
- **Analytics**: Google Analytics tracking (UA-96728449-1)
- **Directory page**: Custom JavaScript-based directory/index view with collapsible tree structure
- **Bootstrap integration**: Bootstrap CSS/JS framework for responsive design

## Jekyll Configuration

Key settings in `_config.yml`:
- Uses Kramdown with GitHub Flavored Markdown
- Rouge syntax highlighter
- Jekyll plugins: paginate, sitemap, feed
- Custom permalink structure: `/:title/`
- Posts collection with custom permalink: `/posts/:path/`

## Development Commands

This site uses GitHub Pages' automatic Jekyll processing with no local build system:
- **Deployment**: Automatic on push to `master` branch via GitHub Pages
- **Local development**: Jekyll would need manual setup (no Gemfile present)
- **Testing changes**: Direct file editing with GitHub's web interface or local Jekyll serve
- **Dependencies**: Bootstrap 3.3.7, jQuery 3.7.1, Tipue Search (all vendored locally)

No build scripts, linting, or test commands - this is a static Jekyll site optimized for GitHub Pages.

## Custom Architecture Components

### Dynamic Directory System
- **Core logic**: `bootstrap/js/directory.js:77-131` implements Tree data structure for hierarchical post navigation
- **DOM generation**: `recurseTree()` function builds collapsible Bootstrap UI from Jekyll post collection  
- **Path parsing**: Extracts directory structure from Jekyll's `site.posts` via `.post-direct` selectors
- **Bootstrap integration**: Uses collapse components for expandable category navigation

### Client-Side Search
- **Tipue Search**: Full-text search across all posts without server dependency
- **Content indexing**: `tipuesearch_content.js` contains indexed post data
- **Modal interface**: Search results display in Bootstrap modal (`default.html:75-85`)
- **Async loading**: Search only loads when modal is triggered

### Performance Optimizations
- **Conditional Mermaid loading**: `default.html:29-47` only loads Mermaid.js when diagrams detected
- **Image preloading**: WebP avatar preloaded for faster rendering (`default.html:16`)
- **Vendored dependencies**: All JS/CSS libraries served locally (no CDN dependencies except Mermaid)

### Template Architecture
- **Multi-language support**: Chinese/English content with consistent navigation
- **Pagination**: 5 posts per page via Jekyll paginate plugin
- **Custom permalinks**: Posts use `/posts/:path/` structure for organized URLs

## Content Guidelines

- Blog posts use YAML front matter with layout specification
- Post filenames follow format: `YYYY-MM-DD-title.md`
- Posts are categorized by subdirectory structure in `_posts/`
- Support for code syntax highlighting and Mermaid diagrams
- Mixed language content (English/Chinese) is common
- Directory page automatically generates from post structure via JavaScript
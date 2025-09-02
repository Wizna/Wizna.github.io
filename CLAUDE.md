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

This is a Jekyll site hosted on GitHub Pages. Since no Gemfile is present, this site relies on GitHub Pages' built-in Jekyll processing:
- **GitHub Pages**: Automatically builds and deploys on push to master branch
- **Local development**: Would require creating a Gemfile with Jekyll and required gems
- **Direct editing**: Files can be edited directly as static HTML/CSS/JS with Jekyll templating

Note: No Gemfile, package.json, or build scripts found - site is designed for GitHub Pages auto-deployment.

## Custom Architecture Components

- **Directory functionality**: `bootstrap/js/directory.js` creates dynamic hierarchical post navigation from Jekyll post collection
- **Search integration**: Tipue Search provides client-side full-text search across all posts
- **Layout structure**: 
  - `default.html`: Main template with search modal, Bootstrap integration, and Mermaid initialization
  - Custom navigation with Blog/Directory/About sections
  - Responsive masthead with site avatar and description
- **Asset organization**: PDFs stored in `assets/summary/` for course materials and documents

## Content Guidelines

- Blog posts use YAML front matter with layout specification
- Post filenames follow format: `YYYY-MM-DD-title.md`
- Posts are categorized by subdirectory structure in `_posts/`
- Support for code syntax highlighting and Mermaid diagrams
- Mixed language content (English/Chinese) is common
- Directory page automatically generates from post structure via JavaScript
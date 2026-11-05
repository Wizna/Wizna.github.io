# Ruiming Huang's Personal Blog

> 一个基于 Jekyll 的个人技术博客，托管在 GitHub Pages。

## 🛠️ 技术栈

- **框架**: Jekyll
- **前端**: Bootstrap, jQuery
- **搜索**: Tipue Search
- **评论**: Disqus
- **部署**: GitHub Pages auto-deployments

## 📱 在线体验

访问 [https://wizna.github.io](https://wizna.github.io) 体验：
- 🏠 **Blog 页**: 最新文章展示，支持分页
- 📂 **Directory 页**: 按分类浏览所有文章
- 🔍 **搜索功能**: 点击搜索图标进行全站搜索
- 📄 **About 页**: 个人介绍和联系方式

## 🤝 联系方式

- **Email**: huangruiming999@gmail.com

## 开发验证

搜索与分类目录的回归测试使用 Node.js 18+：

```sh
npm ci
npm test
```

Push 后由 GitHub Pages 自动构建部署。部署完成后，附带验证线上文章目录与锚点：

```sh
SITE_URL=https://wizna.github.io/ npm test
```

Tipue Search 基于 MIT 许可的 6.0 版本，包含本站的中文搜索、弹窗和安全修复。
截至 2026-11-05，原上游仓库不可访问，cdnjs 收录的最新版本为更旧的 5.0.0；因此保留现有接口并修复本站代码。
`jsdom` 仅用于开发测试，测试文件与 Node.js 依赖不会发布到 GitHub Pages。

# ForTIFAI Website

This directory contains the GitHub Pages website for the ForTIFAI research project.

## 🚀 Quick Start

1. **Fork the repository** to your GitHub account
2. **Enable GitHub Pages** in your repository settings:
   - Go to Settings → Pages
   - Select "Deploy from a branch"
   - Choose "main" branch and "/docs" folder
3. **Update the configuration** in `_config.yml`:
   - Replace `your-username` with your GitHub username
   - Update repository URLs
   - Add your social media links

## 📁 Website Structure

```
docs/
├── index.html          # Main homepage
├── about.html          # Methodology and technical details
├── demo.html           # Interactive demonstration
├── styles.css          # Main stylesheet
├── script.js           # JavaScript functionality
├── _config.yml         # Jekyll configuration
└── README.md           # This file
```

## 🎨 Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Demo**: Real-time model collapse simulation
- **Modern UI**: Clean, professional academic website design
- **Fast Loading**: Optimized CSS and JavaScript
- **SEO Optimized**: Proper meta tags and structured data

## 🔧 Customization

### Updating Content

1. **Homepage**: Edit `index.html`
2. **About Page**: Edit `about.html`
3. **Demo Page**: Edit `demo.html`
4. **Styling**: Edit `styles.css`
5. **Functionality**: Edit `script.js`

### Configuration

Edit `_config.yml` to customize:
- Site metadata
- Navigation menu
- Social links
- Author information
- Benchmark results

### Adding New Pages

1. Create a new HTML file in the `docs/` directory
2. Use the same structure as existing pages
3. Update navigation in `_config.yml`
4. Add links to the main navigation

## 🚀 Deployment

The website is automatically deployed when you push changes to the main branch. GitHub Pages will:

1. Build the site using Jekyll
2. Serve it at `https://your-username.github.io/ForTIFAI`
3. Update automatically when you push changes

## 📱 Mobile Optimization

The website is fully responsive and includes:
- Mobile-first CSS design
- Touch-friendly navigation
- Optimized images and assets
- Fast loading on mobile networks

## 🔍 SEO Features

- Meta tags for search engines
- Open Graph tags for social sharing
- Structured data for rich snippets
- Sitemap generation
- RSS feed support

## 🛠️ Development

### Local Development

1. Install Jekyll: `gem install jekyll bundler`
2. Install dependencies: `bundle install`
3. Run locally: `bundle exec jekyll serve`
4. View at: `http://localhost:4000`

### Testing

- Test all interactive features
- Verify responsive design
- Check cross-browser compatibility
- Validate HTML and CSS

## 📊 Analytics

To add Google Analytics:

1. Get your GA tracking ID
2. Add it to `_config.yml`:
   ```yaml
   google_analytics: GA_TRACKING_ID
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This website is part of the ForTIFAI project and follows the same license as the main repository.

## 🆘 Support

For website-related issues:
- Check the GitHub Issues
- Review the documentation
- Contact the maintainers

For research-related questions:
- See the main repository README
- Check the paper on arXiv
- Contact the authors

---

**Note**: Remember to update all placeholder URLs and usernames in the configuration files before deploying!

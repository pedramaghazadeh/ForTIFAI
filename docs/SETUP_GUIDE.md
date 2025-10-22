# ForTIFAI Website Setup Guide

This guide will help you set up the ForTIFAI website at `https://fortifai.github.io` using GitHub Pages.

## 🎯 Goal

Create a website at `https://fortifai.github.io` that showcases the ForTIFAI research project.

## 📋 Prerequisites

- GitHub account
- Access to create repositories (or organization permissions)
- Basic familiarity with Git

## 🚀 Setup Steps

### Step 1: Create the Repository

1. **Go to GitHub** and create a new repository
2. **Repository name**: `fortifai.github.io` (exactly this name)
3. **Owner**: Your organization or username (e.g., `fortifai`)
4. **Visibility**: Public (required for GitHub Pages)
5. **Initialize**: Don't initialize with README (we'll add files manually)

### Step 2: Upload Website Files

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fortifai/fortifai.github.io.git
   cd fortifai.github.io
   ```

2. **Copy all files** from the `docs/` folder in your ForTIFAI repository to the root of `fortifai.github.io`:
   ```bash
   # Copy all website files to the repository root
   cp -r /path/to/ForTIFAI/docs/* .
   ```

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Initial website setup"
   git push origin main
   ```

### Step 3: Configure GitHub Pages

1. **Go to repository settings**:
   - Navigate to your `fortifai.github.io` repository
   - Click on "Settings" tab

2. **Enable GitHub Pages**:
   - Scroll down to "Pages" section in the left sidebar
   - Under "Source", select "Deploy from a branch"
   - Choose "main" branch
   - Choose "/ (root)" folder
   - Click "Save"

3. **Wait for deployment**:
   - GitHub will automatically build and deploy your site
   - This usually takes 1-2 minutes
   - You'll see a green checkmark when it's ready

### Step 4: Verify Deployment

1. **Visit your site**: `https://fortifai.github.io`
2. **Check all pages**:
   - Homepage: `https://fortifai.github.io/`
   - About: `https://fortifai.github.io/about.html`
   - Demo: `https://fortifai.github.io/demo.html`

## 📁 Repository Structure

After setup, your `fortifai.github.io` repository should look like:

```
fortifai.github.io/
├── index.html          # Main homepage
├── about.html          # Methodology page
├── demo.html           # Interactive demo
├── styles.css          # Stylesheet
├── script.js           # JavaScript
├── _config.yml         # Jekyll configuration
├── README.md           # Documentation
└── SETUP_GUIDE.md      # This guide
```

## 🔧 Configuration Details

The `_config.yml` file has been pre-configured with:
- **URL**: `https://fortifai.github.io`
- **Repository**: `fortifai/fortifai.github.io`
- **Navigation**: Links to all pages
- **Social links**: GitHub, Twitter, LinkedIn
- **SEO settings**: Optimized for search engines

## 🎨 Customization Options

### Update Repository Links
If your main ForTIFAI repository is in a different location, update these URLs in `_config.yml`:
```yaml
code_url: "https://github.com/your-org/ForTIFAI"
```

### Add Analytics
To add Google Analytics, edit `_config.yml`:
```yaml
google_analytics: GA_TRACKING_ID
```

### Custom Domain
To use a custom domain (e.g., `fortifai.org`):
1. Add a `CNAME` file with your domain name
2. Update DNS settings with your domain provider
3. Update the `url` in `_config.yml`

## 🔄 Updating the Website

To update the website:

1. **Make changes** to the files in your local repository
2. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update website content"
   git push origin main
   ```
3. **GitHub Pages** will automatically rebuild and deploy

## 🐛 Troubleshooting

### Common Issues:

1. **Site not loading**:
   - Check if GitHub Pages is enabled in repository settings
   - Verify the repository name is exactly `fortifai.github.io`
   - Wait a few minutes for deployment

2. **Styling issues**:
   - Ensure all CSS files are in the root directory
   - Check browser console for errors
   - Verify file paths in HTML

3. **Interactive features not working**:
   - Check JavaScript console for errors
   - Ensure `script.js` is properly linked
   - Test in different browsers

### Debug Steps:

1. **Check GitHub Pages status**:
   - Go to repository → Actions tab
   - Look for "pages build and deployment" workflow

2. **Verify file structure**:
   - All HTML, CSS, JS files should be in the root
   - No nested folders for main files

3. **Test locally**:
   ```bash
   # Install Jekyll (optional, for local testing)
   gem install jekyll bundler
   bundle install
   bundle exec jekyll serve
   ```

## 📊 Features Included

Your website will have:
- ✅ **Responsive design** (mobile, tablet, desktop)
- ✅ **Interactive demos** (model collapse simulation)
- ✅ **Professional styling** (academic theme)
- ✅ **SEO optimization** (meta tags, structured data)
- ✅ **Fast loading** (optimized assets)
- ✅ **Cross-browser support** (modern browsers)

## 🎉 Success!

Once deployed, your website will be available at:
- **Main site**: https://fortifai.github.io
- **Paper link**: https://arxiv.org/abs/2509.08972
- **Repository**: https://github.com/fortifai/fortifai.github.io

## 📞 Support

If you encounter issues:
1. Check GitHub Pages documentation
2. Review the repository settings
3. Check the Actions tab for build errors
4. Contact GitHub support if needed

---

**Note**: The repository name `fortifai.github.io` is special - GitHub automatically serves it as a Pages site at `https://fortifai.github.io`. This is the cleanest URL for your organization's website.

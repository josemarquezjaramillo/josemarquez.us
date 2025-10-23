# Jose Márquez Jaramillo - Professional Portfolio

A modern, sophisticated Jekyll website showcasing quantitative development work, built with a dark blue aesthetic perfect for the financial services industry.

![Modern Portfolio Screenshot](https://via.placeholder.com/1200x600/0a0e27/3b82f6?text=Modern+Dark+Blue+Portfolio)

## Features

- **Modern Dark Blue Design**: Professional color scheme with deep navy backgrounds and cyan accents
- **Responsive Layout**: Flawless experience across desktop, tablet, and mobile devices
- **Project Showcase**: Dedicated collection for displaying quantitative finance projects
- **Rich Content Support**:
  - LaTeX/MathJax for mathematical equations
  - Code syntax highlighting
  - Embedded charts and visualizations
  - PDF viewers
- **Performance Optimized**: Fast loading with compressed CSS and optimized assets
- **SEO Ready**: Built-in SEO optimization with jekyll-seo-tag
- **GitHub Pages Compatible**: Deploy directly to GitHub Pages with zero configuration

## Quick Start

### Prerequisites

- Ruby 2.7 or higher
- Bundler gem installed
- Git (for version control)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/josemarquezjaramillo/josemarquez.us.git
   cd josemarquez.us
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run the development server**
   ```bash
   bundle exec jekyll serve
   ```

4. **View the site**
   Open your browser to `http://localhost:4000`

The site will automatically rebuild when you make changes to files.

### Building for Production

```bash
bundle exec jekyll build
```

The production-ready site will be in the `_site` directory.

## Site Structure

```
josemarquez.us/
├── _config.yml           # Site configuration
├── _includes/            # Reusable components
│   ├── head.html        # Meta tags, fonts, scripts
│   ├── navigation.html  # Header navigation
│   └── footer.html      # Site footer
├── _layouts/            # Page templates
│   ├── default.html     # Base layout
│   ├── home.html        # Homepage layout
│   ├── page.html        # Standard page layout
│   ├── projects.html    # Projects listing
│   └── project.html     # Individual project
├── _projects/           # Project collection
│   ├── project-1.md
│   ├── project-2.md
│   └── ...
├── _sass/               # SCSS stylesheets
│   ├── _variables.scss  # Design tokens
│   ├── _base.scss       # Base styles
│   ├── _components.scss # Reusable components
│   ├── _navigation.scss # Nav styles
│   ├── _utilities.scss  # Utility classes
│   └── layouts/         # Layout-specific styles
│       ├── _home.scss
│       ├── _projects.scss
│       └── _project-single.scss
├── assets/              # Static assets
│   ├── main.scss        # Main stylesheet entry
│   ├── hero.jpg         # Profile image
│   ├── resume.pdf       # CV/Resume
│   └── images/          # Project images
├── index.md             # Homepage content
├── projects.md          # Projects page
├── contact.md           # Contact page
└── README.md            # This file
```

## Customization Guide

### Updating Site Information

Edit `_config.yml` to update your personal information:

```yaml
title: "Your Name"
tagline: "Your Professional Title"
description: "Brief description for SEO"
author:
  name: "Your Name"
  email: "your.email@example.com"
  linkedin: "your-linkedin-username"
  github: "your-github-username"
```

### Changing Colors

The color scheme is defined in `_sass/_variables.scss`. Key variables:

```scss
// Primary backgrounds
$color-bg-primary: #0a0e27;      // Main background
$color-bg-secondary: #111827;    // Secondary background
$color-bg-elevated: #1a1f3a;     // Cards, modals

// Accent colors
$color-accent-primary: #3b82f6;  // Bright blue
$color-accent-secondary: #06b6d4; // Cyan

// Text colors
$color-text-primary: #e5e7eb;    // Main text
$color-text-secondary: #9ca3af;  // Secondary text
```

### Adding a New Project

1. Create a new Markdown file in `_projects/`:

```bash
_projects/my-new-project.md
```

2. Add front matter and content:

```markdown
---
title: "Project Title"
date: 2024-01-15
category: Quantitative Finance
tags: [Python, Machine Learning, Finance]
featured: true  # Shows badge on card
excerpt: "Brief project description for listing page"
hero_image: /assets/images/project-hero.jpg  # Optional
github_url: https://github.com/user/repo     # Optional
live_url: https://demo.example.com            # Optional
mathjax: true  # Enable if using equations
---

## Project Overview

Your content here...

### Mathematical Equations

Use LaTeX syntax:

$$
E[R_p] = \sum_{i=1}^{n} w_i E[R_i]
$$

### Code Examples

\```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()
\```
```

3. The project will automatically appear on the projects page.

### Updating Homepage Content

Edit `index.md` to update your:
- Professional background
- Education
- Skills and expertise
- Current interests

### Updating Contact Information

Edit `contact.md` to change contact methods and messaging.

### Adding Your Photo

Replace `assets/hero.jpg` with your professional photo. Recommended:
- Format: JPG or PNG
- Size: 800x800px minimum
- Aspect ratio: Square (1:1) or portrait (3:4)
- File size: < 500KB (optimize for web)

### Adding Your Resume

Replace `assets/resume.pdf` with your CV/resume.

## Design System

### Typography

- **Headings**: Inter (weights 600-800)
- **Body**: Inter (weights 400-500)
- **Code**: Fira Code

Loaded from Google Fonts in `_includes/head.html`.

### Spacing Scale

Based on 4px base unit:
- `$space-1`: 4px
- `$space-2`: 8px
- `$space-4`: 16px
- `$space-6`: 24px
- `$space-8`: 32px
- etc.

### Responsive Breakpoints

```scss
$breakpoint-sm: 640px;
$breakpoint-md: 768px;
$breakpoint-lg: 1024px;
$breakpoint-xl: 1280px;
```

### Component Classes

Pre-built components available:

```html
<!-- Buttons -->
<a href="#" class="btn btn--primary">Primary Button</a>
<a href="#" class="btn btn--secondary">Secondary Button</a>

<!-- Cards -->
<div class="card">
  <h3 class="card__title">Card Title</h3>
  <p class="card__description">Description text</p>
</div>

<!-- Tags -->
<span class="tag">Python</span>
<span class="tag tag--primary">Finance</span>

<!-- Grid -->
<div class="grid grid--3-cols">
  <div>Item 1</div>
  <div>Item 2</div>
  <div>Item 3</div>
</div>
```

## Advanced Features

### Embedding Visualizations

Projects support embedded charts and graphs:

```markdown
<div class="project-single__embed project-single__embed--chart">
  <h3 class="project-single__embed-title">Performance Chart</h3>
  <div class="chart-container">
    <!-- Your chart code or iframe -->
    <iframe src="your-chart.html" width="100%" height="400"></iframe>
  </div>
</div>
```

### PDF Embedding

```markdown
<div class="project-single__embed project-single__embed--pdf">
  <h3 class="project-single__embed-title">Research Paper</h3>
  <iframe src="{{ '/assets/papers/research.pdf' | relative_url }}"
          width="100%" height="600px"></iframe>
</div>
```

### Mathematical Equations

LaTeX rendering via MathJax (enable with `mathjax: true` in front matter):

```markdown
Inline equation: $E = mc^2$

Display equation:
$$
\sigma_p^2 = w^T \Sigma w
$$
```

## Deployment

### GitHub Pages (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial site setup"
   git push origin main
   ```

2. **Enable GitHub Pages**
   - Go to repository Settings → Pages
   - Source: Deploy from branch
   - Branch: `main`, folder: `/ (root)`
   - Save

3. **Custom Domain (Optional)**
   - Add `CNAME` file with your domain
   - Configure DNS with your domain provider

The site will be live at `https://yourusername.github.io/repository-name`

### DNS Records for Custom Domain

Point your domain to GitHub Pages:

| Type | Name | Value |
|------|------|-------|
| A    | @    | 185.199.108.153 |
| A    | @    | 185.199.109.153 |
| A    | @    | 185.199.110.153 |
| A    | @    | 185.199.111.153 |
| CNAME| www  | yourusername.github.io |

GitHub will issue an HTTPS certificate automatically.

### Local Testing Before Deployment

```bash
# Build with production settings
JEKYLL_ENV=production bundle exec jekyll build

# Serve the production build
bundle exec jekyll serve --skip-initial-build
```

## Troubleshooting

### Site not building on GitHub Pages?

Check `_config.yml` for GitHub Pages compatibility:
- Only use [supported plugins](https://pages.github.com/versions/)
- Ensure `github-pages` gem is in Gemfile
- Check build logs in GitHub Actions tab

### Styles not loading?

- Clear Jekyll cache: `bundle exec jekyll clean`
- Rebuild: `bundle exec jekyll build`
- Check `assets/main.scss` has front matter (`---`)

### MathJax not rendering?

- Add `mathjax: true` to page front matter
- Ensure MathJax script is in `_includes/head.html`
- Use `$$` for display equations, `$` for inline

### Images not showing?

- Use `| relative_url` filter: `{{ '/assets/image.jpg' | relative_url }}`
- Check image paths are correct
- Ensure images are not in `exclude:` list in `_config.yml`

## Performance Optimization

### Image Optimization

Before adding images:
```bash
# Install ImageMagick or use online tools
convert input.jpg -resize 1200x -quality 85 output.jpg
```

### CSS/JS Minification

Production builds automatically compress CSS (see `_config.yml`):
```yaml
sass:
  style: compressed
```

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Accessibility

The site follows WCAG 2.1 Level AA guidelines:
- Semantic HTML structure
- ARIA labels where appropriate
- Keyboard navigation support
- Sufficient color contrast ratios
- Skip-to-content link
- Responsive text sizing

## License

This website template is free to use for personal portfolios. Attribution appreciated but not required.

## Support & Contact

For questions or issues:
- **Email**: contact@josemarquez.us
- **LinkedIn**: [Jose Márquez Jaramillo](https://www.linkedin.com/in/jose-márquez-jaramillo-b5920535/)
- **GitHub**: [@josemarquezjaramillo](https://github.com/josemarquezjaramillo)

---

Built with Jekyll and modern web standards.

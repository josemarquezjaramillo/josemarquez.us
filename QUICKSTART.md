# Quick Start Guide

Get your modern portfolio site running in 5 minutes!

## Step 1: Install Dependencies

```bash
bundle install
```

This installs Jekyll and all required gems.

## Step 2: Start Development Server

```bash
bundle exec jekyll serve
```

Your site will be available at `http://localhost:4000`

## Step 3: Customize Your Site

### Update Personal Information

Edit `_config.yml`:

```yaml
title: "Your Name"
tagline: "Your Title"
author:
  email: "your.email@example.com"
  linkedin: "your-linkedin-username"
  github: "your-github-username"
```

### Add Your Photo

Replace `assets/hero.jpg` with your professional headshot.

### Add Your Resume

Replace `assets/resume.pdf` with your CV.

### Update Homepage Content

Edit `index.md` to customize your bio, experience, and education.

## Step 4: Add Your First Project

Create `_projects/my-project.md`:

```markdown
---
title: "My Amazing Project"
date: 2024-01-15
category: Finance
tags: [Python, Data Science]
excerpt: "A brief description of what this project does"
---

## Overview

Your project description here...
```

The project will automatically appear on `/projects/`

## Step 5: Deploy to GitHub Pages

```bash
git add .
git commit -m "Personalize portfolio site"
git push origin main
```

Then enable GitHub Pages in your repository settings:
- Settings → Pages
- Source: Deploy from branch `main`

Your site will be live at `https://yourusername.github.io/repository-name`

## Need Help?

See the full [README.md](README.md) for comprehensive documentation.

## Common Commands

```bash
# Start development server
bundle exec jekyll serve

# Build for production
bundle exec jekyll build

# Clean build artifacts
bundle exec jekyll clean

# Update dependencies
bundle update
```

## File Structure Overview

```
Essential files to customize:
├── _config.yml          ← Site settings
├── index.md             ← Homepage content
├── contact.md           ← Contact page
├── _projects/           ← Add your projects here
│   └── project.md
└── assets/
    ├── hero.jpg         ← Your photo
    └── resume.pdf       ← Your resume
```

That's it! You're ready to showcase your quantitative development work.

# Personal website for José Marquez

Built with **Jekyll** and hosted on **GitHub Pages**.

## Local development

```bash
# Install Ruby (>= 3.1) via rbenv or your package manager
gem install bundler
bundle install
bundle exec jekyll serve
```

Site will be served at <http://localhost:4000> and reload on changes.

## DNS records

Point your domain to GitHub Pages:

| Type | Name | Value |
|------|------|-------|
| A    | @    | 185.199.108.153 |
| A    | @    | 185.199.109.153 |
| A    | @    | 185.199.110.153 |
| A    | @    | 185.199.111.153 |
| CNAME| www  | josemarquezjaramillo.github.io |

GitHub will issue an HTTPS certificate automatically.

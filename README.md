## Personal Site (Hugo + hello-friend-ng)

A fast, minimalist personal site built with Hugo and the `hello-friend-ng` theme. This README covers local setup, content authoring, configuration, customization, and deployment to Cloudflare Pages.

### Prerequisites
- Hugo Extended (for SCSS processing)
- Git
- macOS users: Homebrew recommended

Install Hugo on macOS:
```bash
brew install hugo
hugo version
```
Ensure the output shows "+extended".

### Project structure
```
.
├─ archetypes/           # Default front matter templates
├─ content/              # Markdown content (about, posts, projects, etc.)
│  ├─ posts/
│  └─ projects/
├─ layouts/              # Optional custom templates (overrides theme)
├─ static/               # Static assets served as-is (css, img, etc.)
│  ├─ css/override.css   # Your custom CSS overrides
│  └─ img/               # Images (portrait, post images)
├─ themes/hello-friend-ng
├─ resources/            # Hugo pipeline outputs (cache)
├─ public/               # Build output (generated)
└─ hugo.toml             # Site configuration
```

### Quick start (local development)
```bash
# Start dev server with drafts if needed
hugo server -D
# Open http://localhost:1313
```
Changes in `content/`, `layouts/`, or `static/` trigger live reloads.

### Authoring content
Create a new post:
```bash
hugo new posts/my-first-post.md
```
Create a new project page:
```bash
hugo new projects/project1.md
```

Front matter examples:
- Post (dated, uses posts permalink):
```toml
+++
title = "Implementing Linear Regression from Scratch"
date = 2025-08-20T12:00:00Z
tags = ["linear-regression", "AI"]
categories = ["machine-learning"]
draft = false
+++
Your post body here.
```
- Project (simple page):
```toml
+++
title = "Predicting Housing Prices"
date = 2025-08-24
draft = false
+++
Short description, links, screenshots.
```

Tips
- Set `draft = false` to publish.
- Place images under `static/img/` and reference them as `/img/filename.ext` in Markdown.
- Shortcodes from the theme are available; see `themes/hello-friend-ng/layouts/shortcodes/`.

### Configuration (`hugo.toml`)
Key options used here:
```toml
baseURL = "/"                 # Set to your production URL on deploy
canonifyURLs = true
languageCode = "en-us"
theme = "hello-friend-ng"

[pagination]
  pagerSize = 10

[permalinks]
  posts = "/posts/:year/:month/:title/"

[params.portrait]
  path     = "img/profile.jpg" # Or use profile.png if you prefer
  alt      = "My portrait"
  maxWidth = "150px"

[params]
  homeSubtitle = "Your friendly neighbourhood Data Scientist"
  description  = "Portfolio showcasing AI & Data Science projects"
  keywords     = "AI, Data Science, Machine Learning, Portfolio"
  enableThemeToggle = true
  defaultTheme = "auto"
  customCSS = ["css/override.css"]

[[params.social]]
  name = "github"
  url  = "https://github.com/Theani7"
```
Menu entries are configured via `[[menu.main]]` entries. Taxonomies (tags, categories, series) are enabled under `[taxonomies]`.

### Theming and custom CSS
- Theme: `themes/hello-friend-ng`
- Add small style tweaks in `static/css/override.css`.
- For larger layout changes, create files in `layouts/` that mirror the theme’s structure to override specific templates.

### Production build
```bash
hugo --minify
```
The optimized site is emitted to `public/`.

## Deploying to Cloudflare Pages

### 1) Push to GitHub
Initialize the repo (if not already):
```bash
git init
git add .
git commit -m "Initial site"
git branch -M main
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git push -u origin main
```

### 2) Create a Cloudflare Pages project
- In Cloudflare dashboard → Pages → Create project → Connect to Git
- Select your GitHub repository
- Set the following build settings:
  - Framework preset: None
  - Build command: `hugo --minify`
  - Build output directory: `public`
- Environment variables (recommended):
  - `HUGO_VERSION` = the version you use locally (e.g., `0.148.2`)
  - `HUGO_ENV` = `production`

### 3) Set baseURL
Update `baseURL` in `hugo.toml` to your Pages domain once known:
```toml
baseURL = "https://yourname.pages.dev/"
```
If using a custom domain, set it to that final URL instead.

### 4) Trigger a deploy
Push to `main` (or your production branch). Cloudflare builds and deploys automatically.

### Custom domains on Pages
- Add your custom domain in the Pages project → Custom domains
- Configure DNS records as instructed by Cloudflare
- Update `baseURL` to `https://www.yourdomain.com/`

### Common environment variables
- `HUGO_VERSION`: Ensures Pages uses the exact Hugo version you expect
- `HUGO_ENV`: Usually `production`; Hugo can use this to toggle behavior

### Troubleshooting
- Stale files in `public/` (e.g., renamed pages): Run a clean build locally
```bash
hugo --minify --cleanDestinationDir
```
- Broken portrait image: ensure it exists at `static/img/profile.jpg` (or update `params.portrait.path` to `img/profile.png`).
- 404s for new content: confirm `draft = false` and that the section lives under `content/` in the expected path.
- CSS not applying: confirm `static/css/override.css` exists and is referenced in `hugo.toml` (`params.customCSS`).
- Base URL issues: make sure `baseURL` matches your actual deployed domain and `canonifyURLs = true` is set if you prefer absolute URLs.

### Useful commands
```bash
# Start dev server with drafts
hugo server -D

# Production build
hugo --minify

# Clean build (removes stale files in public/)
hugo --minify --cleanDestinationDir
```
# mynewhugosite

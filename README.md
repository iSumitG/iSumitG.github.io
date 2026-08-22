# sumitg.com

Personal website of Sumit Gupta, live at [sumitg.com](http://sumitg.com).

Built with [Jekyll](https://jekyllrb.com/) (Pixyll theme) and published by
GitHub Pages.

## Run locally

Requires Ruby 3.3 (the version GitHub Pages builds with; the pinned
`github-pages` gem does not install on Ruby 4). On macOS:

```sh
brew install ruby@3.3      # keg-only; does not replace your default ruby
./start.sh                 # installs gems on first run, then serves
```

Open <http://localhost:4000>. The site rebuilds and live-reloads on change.

`start.sh` puts Homebrew's `ruby@3.3` on `PATH` automatically. With another
Ruby manager (rbenv, asdf, mise) the version is in `.ruby-version`; run
`bundle install && bundle exec jekyll serve --watch --livereload` instead.

If a build fails with `Invalid US-ASCII character`, the shell has no UTF-8
locale (Ruby Sass needs one): `export LANG=en_US.UTF-8`. `start.sh` does this
for you.

Other useful commands:

```sh
bundle exec jekyll build            # one-off build into _site/
JEKYLL_ENV=production bundle exec jekyll build   # exact GitHub Pages output (minified CSS)
rake post['My post title']          # scaffold _posts/YYYY-MM-DD-my-post-title.md
rake draft['My draft title']        # scaffold _drafts/my-draft-title.md
bundle exec jekyll serve --drafts   # preview drafts too
```

## Deploy

Push to `master`. GitHub Pages builds the site itself with the
[`github-pages`](https://github.com/github/pages-gem) gem (Jekyll 3.10) and
serves it at the custom domain in `CNAME`. No CI config is needed; only the
plugins GitHub Pages whitelists are available.

The `Gemfile` pins the same `github-pages` gem so a local build matches what
GitHub produces.

## Editing content

- Blog posts: Markdown in `_posts/` (kramdown, MathJax enabled).
- Projects, papers, timeline: YAML in `_data/`, rendered by the
  `projects/`, `papers/`, and `about/` pages.
- Site-wide settings and social links: `_config.yml`.

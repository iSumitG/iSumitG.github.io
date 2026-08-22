#!/usr/bin/env bash
# Serve the site locally at http://localhost:4000 (rebuilds + live-reloads on change).
set -euo pipefail
cd "$(dirname "$0")"

# GitHub Pages builds with Ruby 3.3.x; the pinned github-pages gem does not
# install on Ruby 4. Prefer Homebrew's keg-only ruby@3.3 if it is present.
if [ -d /opt/homebrew/opt/ruby@3.3/bin ]; then
  export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
fi

# Ruby Sass needs a UTF-8 locale; fall back to one if the shell has none.
if [ -z "${LC_ALL:-}" ] && [ -z "${LANG:-}" ]; then
  export LANG=en_US.UTF-8
fi

bundle check >/dev/null 2>&1 || bundle install
exec bundle exec jekyll serve --watch --livereload "$@"

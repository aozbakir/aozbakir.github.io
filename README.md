# aozbakir.github.io

Personal academic site for Ali Değer Özbakır, built with [Academic Pages](https://github.com/academicpages/academicpages.github.io) (a Jekyll theme). Served automatically from the `main` branch via GitHub Pages.

## Local preview

The system Ruby is too new for `github-pages`'s native extensions, so this repo uses Homebrew's Ruby 3.1 with gems vendored to `vendor/bundle-ruby31/` (see `.bundle/config`):

```
/opt/homebrew/opt/ruby@3.1/bin/bundle install
/opt/homebrew/opt/ruby@3.1/bin/bundle exec jekyll serve --port 4000
```

If Ruby 3.1 isn't installed: `brew install ruby@3.1`.

If the dev server ever serves a stale page or a wrong permalink after several quick edits, stop it, delete the build cache, and restart:

```
rm -rf _site .jekyll-cache .jekyll-metadata
```

## Editing content

* `_config.yml`: site-wide settings (name, bio, links)
* `_pages/about.md`: homepage bio
* `_pages/cv.md`: CV page
* `_projects/`: one file per project
* `_publications/`: one file per publication
* `_teaching/`: one file per course
* `_data/news.yml`: news timeline (`/news/` page and the homepage preview)

See `CONVENTIONS.md` for the site's writing and visual conventions.

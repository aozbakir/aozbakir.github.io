# Conventions

Style and structure conventions for this site, kept here so they survive across editors and sessions.

## Writing

* No em-dashes anywhere on the site — prose, lists, everything. Use a period, comma, or parenthesis instead.
* Prefer short, direct sentences over long compound ones.
* No PhD-recruitment call to action anywhere (About, Projects, Teaching). There's no internal funding for PhD students, only external — don't imply otherwise.

## Projects (`_projects/`)

* `status:` front matter is `"Completed"` or `"Ongoing"` — rendered as a tag next to the title on the listing page.
* When a project's budget and duration are known, add them in parentheses at the end of the `excerpt:` front matter (what shows on `/projects/`) and again in the matching sentence in the body, e.g. `(€500,000 budget, 20 months)` or `(€2.22M total budget)`.
* `header.teaser` (not `header.image`) for any social-share image — see "Share images" below.

## Listing pages (News, Projects, Publications, Teaching, Blog)

These collections share one timeline visual: a vertical line with a dot per entry, defined once in `_includes/head/custom.html` (`.projects-timeline`, `.teaching-timeline`, `.publications-timeline`, `.blog-timeline`, `.news-timeline`). Date-first layout (date, then title, then body) is handled in `_includes/archive-single.html`. Keep new listing pages consistent with this pattern rather than introducing a new layout.

## Share images (Open Graph / Twitter Card)

Set `header.teaser`, not `header.image`, in a post/project's front matter. `header.image` also triggers the theme's full-width hero banner in `_layouts/single.html` (via `_includes/page__hero.html`), which breaks the sidebar layout. `_includes/seo.html` reads `header.teaser` as an OG/Twitter fallback without that side effect.

## Architecture diagrams (`images/*-architecture.svg`)

Hand-authored SVGs, not exported from a diagramming tool, matching one visual language across MAS4TE, MAI-HOME, and Landslide Hunter:

* `viewBox`, Helvetica/Arial font, one `<marker id="arrow">` triangle arrowhead reused for every line.
* Rounded rects (`rx="6"`), color-coded by role:
  * gray `#f5f5f5` fill / `#999` stroke — infrastructure or physical components
  * orange `#fde3cc` / `#c76b2c` — models or trainable components
  * blue `#dce8f5` / `#4a6fa5` — pipeline/process steps or dashboards
  * purple `#ece3f5` / `#8a5fc7` — stakeholders
* Dashed containers (`stroke-dasharray`) for conceptual groupings (a house, a digital twin, a models frame) rather than a legend.
* Routing: simple, minimal-turn orthogonal paths that exit a box's own edge cleanly. Avoid diagonal lines and avoid paths that graze along an unrelated box's border. A smooth bezier curve is acceptable for a single long-range connection that would otherwise cut across the whole diagram.

## Local dev

See `README.md` for the Ruby 3.1 / vendored-gems setup and the stale-build fix.

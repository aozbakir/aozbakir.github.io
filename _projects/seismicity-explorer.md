---
title: "Seismicity Explorer"
collection: projects
permalink: /project/seismicity-explorer
excerpt: "Full-stack earthquake hazard modeling and event-set simulation platform, built solo at Temblor Inc."
date: 2022-01-01
order: 7
venue: "Temblor Inc."
status: "Completed"
tech: ["Python", "Flask", "PostGIS", "MapLibre", "Hazard Modeling"]
---

A full-stack earthquake hazard modeling platform that blends two complementary forecasting methods into one interactive tool: smoothed seismicity (adaptive, self-calibrating anisotropic Gaussian kernel density estimation over instrumental earthquake catalogs) and a tectonic strain-rate forecast following Bird & Liu's SHIFT/GEAR1 approach, combinable via a single blend weight into a hybrid hazard model. Lets users explore rate-density grids and magnitude-frequency fits for any region, then drive stochastic event-set simulations from the result: the same class of primitive (synthetic catalogs, return-period recurrence, Gutenberg-Richter magnitude scaling) that catastrophe-modeling vendors sell to the reinsurance industry for pricing and capital allocation.

![Seismicity Explorer's blend view: a rate-density heatmap over California, with magnitude-frequency statistics and incremental/cumulative MFD fit plots alongside](/images/seismicity-explorer-ui.png)

**Technology:** Flask, PostGIS, and MapLibre. The backend follows a hexagonal, ports-and-adapters architecture with a dependency-free domain core, validated by 718 tests, and a multi-layer caching system (in-memory LRU, NetCDF-backed persistence, spatial indices) that turns 10–30 second live computations into millisecond responses on repeat queries.

**My role:** Built the whole platform solo.

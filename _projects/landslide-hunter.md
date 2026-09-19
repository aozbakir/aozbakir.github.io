---
title: "Landslide Hunter"
collection: projects
permalink: /project/landslide-hunter
excerpt: "Funded by NWO (€80,000 budget, 1 year)."
date: 2024-06-01
order: 4
venue: "NWO-funded; Open Universiteit and University of Twente"
status: "Completed"
tech: ["Python", "Deep Learning", "Earth Observation", "Image Transformers", "CNN"]
---

A fully automated Earth-observation platform for rapid landslide detection and mapping (2024–2025), extending time-series and index-based heuristics to optical satellite imagery to map landslides even in semi-cloudy conditions. Built during a postdoctoral position at Open Universiteit and University of Twente. Total budget €80,000 over 1 year.

![Landslide Hunter's campaign pipeline: scene discovery over Sentinel-2 via STAC, patch download, traditional and deep-learning model inference with a confidence-based re-acquisition loop, then pre/post-event time-series change detection into a results map, backed by PostgreSQL, UNet weights, and a Celery/Redis task queue](/images/landslide-hunter-architecture.svg)

Has produced 3 conference papers: 2 EGU General Assembly abstracts and 1 peer-reviewed paper (Living Planet Symposium 2025). See [Publications](/publications/).

![Model Sherpa's campaign view for Landslide Hunter: a results map over two MGRS tiles alongside acquisition, model-run, and confidence progress for an active monitoring campaign](/images/landslide-hunter-results-map.png)

**Technology:** Python campaign-orchestration platform (FastAPI, Celery/Redis, PostgreSQL) discovering Sentinel-2 scenes via STAC, patch-based tiling of GeoTIFF imagery for inference across traditional (NDVI, BSI) and deep-learning (UNet) models, with iterative acquisition governed by per-tile confidence-based stopping criteria and pre/post-event time-series change detection.

**My role:** Implemented the core functionalities: model inference (traditional and deep-learning) and iterative mosaicking with uncertainty-based stopping criteria.

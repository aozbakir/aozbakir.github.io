---
title: "Landslide Hunter"
collection: projects
permalink: /project/landslide-hunter
excerpt: "Funded by NWO."
date: 2024-06-01
order: 4
venue: "NWO-funded; Open Universiteit and University of Twente"
tech: ["Python", "Deep Learning", "Earth Observation"]
---

A fully automated Earth-observation platform for rapid landslide detection and mapping (2024–2025), extending time-series and index-based heuristics to optical satellite imagery to map landslides even in semi-cloudy conditions. Built during a postdoctoral position at Open Universiteit and University of Twente.

Has produced 3 conference papers — 2 EGU General Assembly abstracts and 1 peer-reviewed paper (Living Planet Symposium 2025) — see [Publications](/publications/).

**Technology:** Python campaign-orchestration platform (FastAPI, Celery/Redis, PostgreSQL) discovering Sentinel-2 scenes via STAC, patch-based tiling of GeoTIFF imagery for inference across traditional (NDVI, BSI) and deep-learning (UNet) models, with iterative acquisition governed by per-tile confidence-based stopping criteria and pre/post-event time-series change detection.

**My role:** Implemented the core functionalities — model inference (traditional and deep-learning) and iterative mosaicking with uncertainty-based stopping criteria.

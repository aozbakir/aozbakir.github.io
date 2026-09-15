---
title: "Green Innovation Extraction"
collection: projects
permalink: /project/green-innovation-extraction
excerpt: "Funded by GRINS, in collaboration with the University of Verona (Cristina Vlorio et al.)."
date: 2026-04-01
order: 2
venue: "GRINS (Growing Resilient, INclusive and Sustainable); in collaboration with University of Verona"
tech: ["Python", "LLM", "NLP"]
---

A pipeline for extracting and classifying "green innovation" claims and green product labeling from corporate annual reports, scanning and analyzing 40+ Italian listed companies' annual reports. Combines automated environmental-relevance filtering (BERT-based scoring) with LLM-driven extraction and multi-label classification against established green/circular-economy taxonomies, with full database-backed traceability from source text to finding.

In collaboration with the University of Verona (Cristina Vlorio et al.), funded by GRINS.

**Technology:** Python pipeline with SQLAlchemy/Alembic-backed persistence and a Chroma vector store, combining transformer-based environmental-relevance scoring (EnvironmentalBERT/ClimateBERT) with LLM-driven extraction (OpenAI/Anthropic APIs) and multi-label classification of corporate disclosures, built with a domain-driven, onion-architecture design.

**My role:** Tech lead for GRINS; wrote the whole pipeline end to end — the full extraction, classification, and traceability process.

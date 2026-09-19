---
title: "MAI-HOME"
collection: projects
permalink: /project/mai-home
excerpt: "Funded via Interreg Flanders–Netherlands (€4M total budget, 3 years). Coordinated by Open Universiteit with five housing corporations, SMEs, and social/knowledge institutions."
date: 2023-09-01
order: 3
venue: "Interreg Flanders–Netherlands; coordinated by Open Universiteit"
status: "Completed"
tech: ["Python", "LSTM", "Transformers", "GNN"]
---

An Interreg Flanders–Netherlands project (started September 2023) combating energy poverty and reducing CO₂ emissions from housing across Dutch and Belgian Limburg. Builds on the earlier GO-KIT pilot. Combines sensor-based monitoring of living behavior with occupancy and environmental forecasting models that generate personalized energy-saving guidance, delivered through gamification and social learning, on the premise that renovation alone doesn't sustain energy savings once residents' motivation fades. Also runs educational workshops and a MOOC for housing-corporation and municipal staff. Coordinated by Open Universiteit with five housing corporations, SMEs, and social/knowledge institutions. Total budget €4M over 3 years.

![MAI-HOME architecture: sensors and heating system inside the house connect via a LoRa gateway to a digital twin (server, database, and occupancy/thermal-comfort/energy-use models), which schedules heating back to the house and feeds an energy dashboard that housing corps and tenants act on](/images/maihome-architecture.svg)

Has produced peer-reviewed papers in *Energy and Buildings* and at ACM SAC, and the HOMESENSE multisensor dataset (accepted, *Scientific Data*). See [Publications](/publications/).

**Technology:** Python ML pipeline with modular data-download, processing, training, and inference stages; LSTM and Transformer model architectures for occupancy and energy-use inference from sparse IoT sensor data.

**My role:** Joined in 2025 as work package lead, responsible for delivering the database, the models, and the energy dashboard. Also responsible for app design; implementation was carried out by a third party.

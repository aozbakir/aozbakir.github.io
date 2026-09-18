---
title: "GO-KIT"
collection: projects
permalink: /project/go-kit
excerpt: "Coordinated by Open Universiteit with data partner Calculus, housing corporations Weller Wonen, Wonen Zuid, and Cordium, and installer Habenu Van de Kreeke."
date: 2022-10-01
order: 6
venue: "Cross-border NL–BE Limburg project; coordinated by Open Universiteit"
status: "Completed"
tech: ["Python", "LSTM", "scikit-learn"]
---

A pilot (2022–2024) deploying AI-driven smart-home sensors in 16 social housing units across Dutch and Belgian Limburg — 10 homes in Heerlen, 5 in Bilzen/Hoeselt, and 1 backup — aiming to cut energy use and address energy poverty without requiring residents to change their behavior. Sensors track occupancy, humidity, appliance activity, and energy/water use; an LSTM+CNN model (developed at Open Universiteit, later adopted as the common baseline in the graph neural network occupancy-prediction work) automatically adjusts heating, lighting, and water systems and gives personalized recommendations. Served as the pilot for MAI-HOME, the later Interreg Flanders–Netherlands project that built on GO-KIT to add the behavioral-change side through psychology, gamification, and energy coaches.

Coordinated by Open Universiteit (CAROU) with data partner Calculus, the AI hub at Brightlands Smart Services Campus, housing corporations Weller Wonen, Wonen Zuid, and Cordium, installer Habenu Van de Kreeke, and the Construction Confederation (Belgian Limburg).

**Technology:** Python data pipeline (pandas, scikit-learn) integrating the Calculus IoT API into a sensor database, with data cleaning and Isolation-Forest-based anomaly detection across occupancy, environmental, and utility sensors.

**My role:** Joined in 2023 as a postdoctoral researcher; executed the full technical implementation, including running LSTM networks in the test house.

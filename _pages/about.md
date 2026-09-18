---
permalink: /
title: "Ali Değer Özbakır"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

I am an Assistant Professor of Computer Science / AI at [Open Universiteit](https://www.ou.nl/), Heerlen, Netherlands, where I co-developed the MSc course Software Engineering and AI and teach Time Series Analysis and Forecasting.

My research centers on machine learning for time series: occupancy inference and energy optimization from sparse IoT sensor data, using RNNs, Transformers, and graph neural networks. I also work on multi-agent systems for peer-to-peer energy trading, including blockchain-based smart contracts for battery storage markets.

I'm fascinated by epistemic agency: the capacity to ask questions, evaluate evidence, and revise beliefs within shared rules, what Wittgenstein called language games. Chatbot conversation complicates this in a specific way: it has the surface form of dialogue, what Habermas called pseudo-communication, without the mutual understanding real dialogue requires. What draws me to generative AI is how it raises these questions at scale, and how easily the resulting failures get framed as individual rather than institutional.

Before moving into computer science, I trained as a geophysicist (PhD, Utrecht University), working on GNSS network analysis, earthquake hazard modeling, and Earth observation. That background still shapes how I think about probabilistic modeling and messy, real-world sensor data.

A full list of positions, teaching, and publications is on my [CV](/cv/), [Teaching](/teaching/), and [Publications](/publications/) pages.

<div class="cv-skill-group"><span class="cv-tag">Time-Series ML</span><span class="cv-tag">Occupancy Inference</span><span class="cv-tag">Multi-Agent Systems</span><span class="cv-tag">Energy Trading</span><span class="cv-tag">Blockchain/Smart Contracts</span><span class="cv-tag">AI Ethics</span><span class="cv-tag">Disinformation</span></div>

## Recent news

<div class="news-timeline">
{% assign recent_news = site.data.news | sort: "date" | reverse %}
{% for item in recent_news limit: 3 %}
  <div class="timeline-item">
    <div class="news-date">{{ item.date | date: "%b %-d, %Y" }}</div>
    <p class="news-text">{{ item.text | markdownify | remove: "<p>" | remove: "</p>" }}</p>
  </div>
{% endfor %}
</div>

[See all news →](/news/)

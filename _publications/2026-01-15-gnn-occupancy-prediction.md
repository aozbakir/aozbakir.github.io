---
title: "Improving indoor occupancy prediction using graph neural networks and positional encodings"
collection: publications
category: manuscripts
permalink: /publication/2026-gnn-occupancy-prediction
excerpt:
date: 2026-01-15
venue: 'Energy and Buildings, 370(Part A), 118077'
paperurl:
status: Published
citation: 'Sheng, Y., Özbakır, A.D., Iren, D., Maathuis, C., &amp; Bromuri, S. (2026). "Improving indoor occupancy prediction using graph neural networks and positional encodings." <i>Energy and Buildings</i>, 370(Part A), 118077.'
---

**Abstract:** Managing residential energy efficiently without compromising comfort is challenging. The key difficulty lies in predicting occupancy in real time. Intelligent control strategies are widely used for residential energy management. As sensors become increasingly common in homes, occupancy prediction has gained significant attention as one such approach. Traditional occupancy prediction methods analyze temporal patterns within individual rooms but overlook spatial connectivity. This limitation reduces the accuracy of occupancy-forecasting methods. Incorporating spatial information could improve prediction accuracy and enhance the effectiveness of occupancy-based energy management. One promising approach to represent such spatial information is through the use of embeddings, which encode spatial connectivity in a data-driven vector space. Another complementary method is to represent the building layout as a graph, where nodes correspond to rooms and edges capture spatial connectivity. In this study, we evaluate multiple occupancy prediction architectures, including Convolutional Neural Networks (CNN) and Long Short-Term Memory networks (LSTM). These models are further enhanced by positional encoding (PE) and Graph Neural Networks (GNN) that explicitly encode spatial connectivity among rooms. Results indicate that spatial embeddings improve predictive performance, with graph-based methods providing the highest performance. Among the evaluated models, GNN + CNN achieved the highest average AUC of 0.838, indicating the strongest threshold-independent discrimination. GNN + PE + CNN achieved strong threshold-dependent performance, with an average accuracy of 0.814 and an average F1-score of 0.830. Considering predictive performance together with computational cost, LSTM + PE is identified as a promising lightweight candidate for practical heating-control applications.

**Code:** [GitHub repository](https://github.com/YuSylvan/Improving-Indoor-Occupancy-Prediction-using-Graph-Neural-Networks-and-Positional-Encodings)

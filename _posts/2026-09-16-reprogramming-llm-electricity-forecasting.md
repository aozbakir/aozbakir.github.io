---
title: "Reprogramming a General-Purpose LLM for Electricity Demand Forecasting"
excerpt: "Reprogramming a general-purpose LLM to forecast electricity demand, instead of building yet another specialized time-series model. Here's the case for it, and how it currently stacks up against Chronos."
tags: ["Time-Series ML", "LLMs", "Energy"]
---

Residential electricity demand is becoming harder to predict. Solar panels, batteries, heat pumps, EV chargers: a household now draws from the grid and feeds back into it on a much shorter cycle than older forecasting methods were built for. Day-ahead forecasts feed directly into how grid operators balance supply and demand. They also feed into how peer-to-peer energy markets, like the one I work on in [MAS4TE](/project/mas4te), price and settle trades. Better forecasts at the household level are genuinely useful.

Until recently, the standard toolkit here was classical: seasonal-naive baselines, ARIMA-family models, gradient-boosted trees, later RNNs and Transformers trained from scratch on each dataset. That's shifted in the last two years. A new generation of *time-series foundation models*, [Chronos](https://arxiv.org/abs/2403.07815), [TimesFM](https://arxiv.org/abs/2310.10688), [Moirai](https://arxiv.org/abs/2402.02592), and others, are pretrained on huge, diverse collections of time series. They generalize to a new problem zero-shot, the same way GPT-style models generalize across text tasks without retraining. Chronos in particular is a strong baseline. Hard to beat.

But there's a second idea running alongside that trend. I find it more interesting. Instead of training a new model purely on time-series data, *reprogram* an existing general-purpose LLM, one already pretrained on huge amounts of text, to also handle numerical time series. [Time-LLM](https://arxiv.org/abs/2310.01728) (Jin et al., ICLR 2024) made this concrete. Patch the series: cut it into short, fixed-length chunks. Project each patch into something that looks like the LLM's own token embeddings, via a learned cross-attention layer over prototype vectors clustered from the LLM's own frozen vocabulary. Keep the backbone frozen. Train only a small adapter.

Why bother, if a purpose-built time-series model already works well? Because a general-purpose LLM doesn't stop being a language model just because you've taught it a second trick. Chronos produces a forecast and nothing else. It has no language understanding at all. It can't take a natural-language question about the data. It can't explain a prediction. It can't reason about context described in words. A reprogrammed LLM, in principle, still can. That's worth exploring, even if raw accuracy doesn't win outright.

`loadcast` is my own attempt at this. Reprogramming [EuroLLM-1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B), a frozen, general-purpose multilingual LLM never trained for time series, into a day-ahead electricity demand forecaster. LoRA is that adapter. The backbone's own weights never change. LoRA adds small trainable matrices next to a few of them instead. Only a tiny fraction of the 1.7B parameters actually get updated. That's the light-touch fine-tuning. Training and evaluation both run on [Low Carbon London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london), half-hourly smart-meter readings from over 5,500 London households.

![loadcast architecture: kWh history and covariates are patched and embedded, projected into the frozen LLM's token space via a reprogramming cross-attention layer, passed through frozen EuroLLM with LoRA adapters, and decoded into a day-ahead forecast](/images/loadcast-architecture.svg)

That's the forward pass. Here's the training loop around it, logging and data loading stripped out:

```python
for epoch in range(config.training.epochs):
    for batch in loader:
        optimizer.zero_grad()
        forecast = model(batch)

        # loss in RevIN-normalized space: computing it on raw kWh would let
        # high-consumption households dominate the gradient
        context_stats = RevIN()
        context_stats.normalize(batch["target"])
        forecast_normed = (forecast - context_stats.mean) / context_stats.std
        future_normed = (batch["future"] - context_stats.mean) / context_stats.std

        # huber, not mse: caps how much a single noisy window can swing the loss
        loss = torch.nn.functional.huber_loss(forecast_normed, future_normed, delta=1.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, config.training.grad_clip_norm)
        optimizer.step()
        scheduler.step()

    holdout_mase = evaluate_holdout_mase(model, holdout_loader, holdout_building_series, config)
    save_checkpoint(model, config, run_name)
    if holdout_mase < best_holdout_mase:
        best_holdout_mase = holdout_mase
        save_checkpoint(model, config, f"{run_name}_best")
```

RevIN, reversible instance normalization, rescales each window by its own mean and standard deviation before the model ever sees it, then undoes that rescaling on the output. The loss is computed in that normalized space, not raw kWh, so no household's absolute consumption scale dominates the gradient. It's Huber, not MSE, so one noisy window can't swing the whole batch's loss.

The split is by household, not by time. 900 training households, 30 held out. Stratified by ACORN group (a UK classification of households by socioeconomic and lifestyle type) and tariff type, drawn once with a fixed seed. No k-fold cross-validation. Holdout households never appear in training at all, so this measures generalization to buildings the model has never seen, not just unseen time windows from familiar ones.

Each forecast uses 7 days of history to predict the next day, half-hour by half-hour. Evaluation runs on a fixed window stride across the holdout set: 5,944 windows total. Training took about 100 minutes on an M4 MacBook Pro. No GPU cluster.

MASE, mean absolute scaled error, scores a forecast against a naive same-time-yesterday repeat. Below 1.0 means beating that baseline.

Overfitting is checked by evaluating the same model on a same-size sample of training households instead of the holdout set. That scores MASE 0.844, close to the holdout's 0.855. That's a small enough gap to rule out serious overfitting. Holdout MASE is also tracked every epoch during training, and the best checkpoint isn't always the last one. Here it peaked at epoch 6 of 8, then ticked back up slightly.

The comparison is in. Chronos here is `chronos-bolt-small`, 48M parameters, T5-based, zero-shot. Auto-ARIMA (via [statsforecast](https://github.com/Nixtla/statsforecast)) fits a seasonal order per household, then refits it for every evaluation window. LightGBM is a single gradient-boosted-trees model, trained once across all training households on lagged and calendar features, the one classical baseline that actually learns from the same training data LoadCast does. Seasonal-naive just repeats the value from exactly one day earlier.

| Method | MASE |
|---|---|
| Chronos (zero-shot) | 0.755 |
| LoadCast (covariates off) | 0.855 |
| LoadCast (covariates on) | 0.855 |
| Auto-ARIMA | 0.981 |
| LightGBM | 0.992 |
| Seasonal-naive | 1.003 |

Chronos wins. LoadCast hasn't closed the gap yet. It beats every classical baseline. Auto-ARIMA, LightGBM, and seasonal-naive all cluster within 0.02 MASE of each other, essentially tied. LoadCast holds a real margin over all three. Covariates make almost no difference, on or off, despite six separate encoders and learned gates for all of them. The one thing that's actually helped is training on more households. 150, then 450, then 900. Each step improved holdout MASE a bit further.

The table only shows averages. Here's what six of those held-out households actually look like, forecast window by forecast window:

![Forecast comparison across six held-out households: LoadCast, Chronos, Auto-ARIMA, LightGBM, and seasonal-naive](/images/loadcast-forecast-comparison.png)

LoadCast and Chronos track closely on most of these. Both smooth over the sharpest spikes in actual demand. On one low-consumption household, LoadCast and LightGBM both get visibly noisier than Chronos and Auto-ARIMA, which stay flat.

So the general-purpose LLM doesn't win yet. It's not obviously the wrong bet either. Still ahead of the classical toolkit, with the clearest lever so far being more data, not a smarter architecture. Whether reprogramming ever closes the gap to a purpose-built time-series model, or whether the real payoff is the language capability Chronos structurally can't have, is still open.

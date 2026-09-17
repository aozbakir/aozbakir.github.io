---
title: "Reprogramming a General-Purpose LLM for Electricity Demand Forecasting"
excerpt: "Reprogramming a general-purpose LLM to forecast electricity demand, instead of building yet another specialized time-series model. Here's the case for it, and how it stacks up against seven other methods, from a naive baseline to a genuine time-series foundation model."
tags: ["Time-Series ML", "LLMs", "Energy"]
---

Residential electricity demand is becoming harder to predict. Solar panels, batteries, heat pumps, EV chargers: a household now draws from the grid and feeds back into it on a much shorter cycle than older forecasting methods were built for. Day-ahead forecasts feed directly into how grid operators balance supply and demand. They also feed into how peer-to-peer energy markets, like the one I work on in [MAS4TE](/project/mas4te), price and settle trades. Better forecasts at the household level are genuinely useful.

Until recently, the standard toolkit here was classical: seasonal-naive baselines, ARIMA-family models, gradient-boosted trees, later RNNs and Transformers trained from scratch on each dataset. That's shifted in the last two years. A new generation of *time-series foundation models*, [Chronos](https://arxiv.org/abs/2403.07815), [TimesFM](https://arxiv.org/abs/2310.10688), [Moirai](https://arxiv.org/abs/2402.02592), and others, are pretrained on huge, diverse collections of time series. They generalize to a new problem zero-shot, the same way GPT-style models generalize across text tasks without retraining. Chronos in particular is a strong baseline. Hard to beat.

But there's a second idea running alongside that trend. I find it more interesting. Instead of training a new model purely on time-series data, *reprogram* an existing general-purpose LLM, one already pretrained on huge amounts of text, to also handle numerical time series. [Time-LLM](https://arxiv.org/abs/2310.01728) (Jin et al., ICLR 2024) made this concrete. Patch the series: cut it into short, fixed-length chunks. Project each patch into something that looks like the LLM's own token embeddings, via a learned cross-attention layer over prototype vectors clustered from the LLM's own frozen vocabulary. Keep the backbone frozen. Train only a small adapter.

Why bother, if a purpose-built time-series model already works well? Because a general-purpose LLM doesn't stop being a language model just because you've taught it a second trick. Chronos produces a forecast and nothing else. It has no language understanding at all. It can't take a natural-language question about the data. It can't explain a prediction. It can't reason about context described in words. A reprogrammed LLM, in principle, still can. Whether it actually does is worth testing, not just assuming, and worth it even if raw accuracy doesn't win outright.

`loadcast` is my own attempt at this. Reprogramming [EuroLLM-1.7B](https://huggingface.co/utter-project/EuroLLM-1.7B), a frozen, general-purpose multilingual LLM never trained for time series, into a day-ahead electricity demand forecaster. At 1.7B parameters, it's small enough to fine-tune on a laptop. Time-LLM's own ablations show reprogramming a bigger backbone forecasts better, so this trades away some of that headroom for running without a GPU cluster. LoRA is that adapter. The backbone's own weights never change. LoRA adds small trainable matrices next to a few of them instead. Only a tiny fraction of the 1.7B parameters actually get updated. That's the light-touch fine-tuning. Training and evaluation both run on [Low Carbon London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london), half-hourly smart-meter readings from over 5,500 London households.

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

The comparison spans a real ladder of approaches, not just a flat list. Seasonal-naive is the baseline. LightGBM is classical ML: a single gradient-boosted-trees model, trained once across all training households on lagged and calendar features. Auto-ARIMA is classical too, but statistical rather than learned: a seasonal order fit per household, refit for every evaluation window. Above that, deep learning trained from scratch on this data: an LSTM and a dilated CNN, both fed the raw sequence directly, sized conventionally for their architecture family rather than parameter-matched to LoadCast. Matching capacity exactly made both impractically slow: the parameter-matched LSTM hadn't finished a single epoch after 30+ minutes, against 88 seconds for the equivalently-sized Transformer, and the parameter-matched CNN stalled the same way after 15+ minutes, against a 2-minute total run for its smaller replacement. Above that, a from-scratch Transformer, patch-based like LoadCast, parameter-matched to LoadCast's own trainable count (19.3M against 18.0M) to isolate whether pretraining is earning its keep: a modern architecture, but no pretraining at all. Chronos (`chronos-bolt-small`, 48M parameters, T5-based) is modern in a different sense, a genuine time-series foundation model, pretrained at scale, zero-shot. LoadCast is the last rung: fine-tuning a general-purpose LLM pretrained on text, not time series.

| Method | Category | MASE | MAE (kWh) |
|---|---|---|---|
| Chronos (zero-shot) | Foundation model | 0.755 | 0.092 |
| Scratch Transformer | Modern (from scratch) | 0.809 | 0.100 |
| LSTM | Deep learning | 0.851 | 0.106 |
| LoadCast | Fine-tuned general LLM | 0.855 | 0.106 |
| CNN | Deep learning | 0.860 | 0.108 |
| Auto-ARIMA | Classical (statistical) | 0.981 | 0.118 |
| LightGBM | Classical (ML) | 0.992 | 0.119 |
| Seasonal-naive | Baseline | 1.003 | 0.120 |

Chronos wins, though not as cleanly as the ranking alone suggests. Its gap to the next-best method is 0.054 MASE, only slightly larger than the 0.051 MASE spread across everything else in the deep-learning cluster below it. LoadCast hasn't closed the gap to Chronos, by about 0.014 kWh per half-hour reading on average. Small per household; whether it matters at grid scale, aggregated across thousands of day-ahead schedules, isn't something this project measures. Below Chronos, the picture is flatter than expected. LoadCast, the scratch Transformer, the LSTM, and the CNN all land within that same 0.051 MASE of each other, a tight cluster of everything that's actually deep learning on this data. LoadCast sits in the middle of it, not ahead, not behind, despite being the only one built on a pretrained backbone, and the only one that took anywhere near as long to train: 100 minutes, against 2 to 12 for the others. None of the from-scratch baselines got their own hyperparameter search either. They inherited LoadCast's own learning rate and schedule, tuned for a frozen-backbone-plus-LoRA setup specifically, so a properly-tuned LSTM or Transformer might do better still. This cluster is a floor for them, not a ceiling. Every deep-learning method still clears the classical toolkit by a real margin: Auto-ARIMA, LightGBM, and seasonal-naive cluster separately, within 0.03 MASE of each other, essentially tied. The one thing that's helped LoadCast specifically is training on more households. 150, then 450, then 900. Each step improved holdout MASE a bit further. Whether more data alone would ever separate it from the rest of the deep-learning cluster, or close the gap to Chronos, isn't tested here. Every number here is also from a single training run, no repeated seeds, so the exact ranking within a few hundredths of MASE is less certain than it looks. The broad shape, Chronos ahead, everything else clustered, classical trailing, is the part worth trusting.

Covariates make almost no difference, on or off. Checking the trained gates explains why: temperature, holiday, and calendar each start at a small, deliberately gentle contribution, a sigmoid-scaled gate initialized to let through about 12% of the signal, and after the full run, every one of them is still sitting within a couple of percentage points of that same starting value. They never learned to open up. Whether that's a training-signal problem, a single scalar gate is a narrow channel for gradient to push through, or these covariates genuinely don't carry much the model can't already infer from recent history, is an open question this doesn't answer yet.

The table only shows averages. Here's what six of those held-out households actually look like, forecast window by forecast window:

![Forecast comparison across six held-out households: LoadCast, Chronos, Auto-ARIMA, LightGBM, the scratch CNN/LSTM/Transformer, and seasonal-naive](/images/loadcast-forecast-comparison.png)

LoadCast and Chronos track closely on most of these. Both smooth over the sharpest spikes in actual demand. The six panels aren't interchangeable profiles, though. Three households are mostly flat with occasional large spikes, and every method, LoadCast included, misses at least one of those spikes outright, less a modeling gap than genuinely unpredictable behavior. One household's single spike, by contrast, is caught closely by most methods, including seasonal-naive, suggesting a more routine, learnable pattern behind it, though the scratch LSTM alone flags a false alarm about ten steps early that nothing else shares. On the low-consumption, high-frequency-noise household, every method except Chronos and Auto-ARIMA chases the noise instead of smoothing over it. It's a tentative read across six windows, not a systematic breakdown by consumption type, but a reminder that one averaged MASE number hides real variation in how, and when, these methods actually fail.

I tested the language-capability argument directly, in a small way. LoadCast itself can't answer anything in words: reprogramming never loads a text-generation head, so the forecast can't be phrased in language by that model at all. Pairing it with a separate, ordinarily instruction-tuned sibling model does work, technically. But the language side is currently weak. Left to reason freely over the numbers, even a well-prompted small instruct model hallucinated readily. A working version needed its role narrowed to paraphrasing a fact computed in advance, checked afterward for invented numbers. An initial exercise, not a verdict on the idea.

Accuracy isn't the only axis that matters, and on the others the ranking flips. Chronos and seasonal-naive need zero training, ever. Point either at a new household's recent history and it just works, the same one-shot advantage that makes foundation models attractive regardless of raw accuracy. Auto-ARIMA is the opposite extreme: the only method here that doesn't generalize across households at all. It needs a fresh order search and refit for every single one. Everything else, LightGBM, the CNN, the LSTM, the scratch Transformer, LoadCast, is a global model that generalizes to new households without refitting, but each needs an initial training pass, and eventually a repeat of it: about 2 minutes for the CNN, 20 for LightGBM, 100 for LoadCast, on this hardware. LoadCast sits at the expensive end of that range for a result that doesn't beat the cheaper alternatives. Explainability tells a similar story. Classical methods are interpretable by construction: Auto-ARIMA's coefficients, LightGBM's feature importances. The from-scratch deep nets are moderately so; attention over raw patches, LSTM gate dynamics, and CNN receptive fields all have mature interpretability tooling built for them. LoadCast is the least legible of any of them. Its forecast arrives via reprogramming into a frozen 1.7B-parameter LLM's embedding space, a genuinely more exotic mechanism, with far less interpretability tooling built for it than any simpler architecture here. That's a qualitative judgment, not something this project measured directly, but it's a real cost worth naming alongside the numbers above.

So the general-purpose LLM doesn't win yet, and the accuracy case for it just got harder to make. LoadCast doesn't stand out from a cluster of from-scratch deep-learning alternatives, an LSTM, a CNN, a plain Transformer, that took a fraction of the training time and, in two of three cases, a fraction of the parameters too. Whatever EuroLLM's pretrained weights contribute here, it isn't separating LoadCast from models built to do nothing but this one task. There are two open questions now, not one: whether reprogramming ever earns back the accuracy a simpler model gets for free, and whether the language capability, tested above and still rough, is real or just a promise the architecture makes but hasn't kept.

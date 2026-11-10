# Experiment 3: training dynamics

Run: `checkpoints\demo_english\log.csv`.

![dynamics](exp3_dynamics.png)

| quantity | value |
|---|---|
| min validation loss | 1.6278 at step 2500 |
| train loss at that step | 1.4237 |
| final train / val loss | 1.4237 / 1.6278 (step 2500) |
| first step with val - train > 0.05 | 500 |
| gap (val - train) at best step / final | 0.2041 / 0.2041 |

**No clear overfitting within this run.** Validation loss never rose more than 0.02 above its minimum (1.6278 at step 2500); it ended at 1.6278. Train more steps, remove dropout or use a bigger model to reach the overfitting regime.

How to read it: before the minimum, both curves fall together (the model is learning general structure). Afterwards the train curve keeps falling while validation turns up: the model is memorising training text. Regularisation (dropout, weight decay), more data, or early stopping at the best checkpoint (`ckpt_best.pt`) address it.

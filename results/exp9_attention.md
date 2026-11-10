# Experiment 9: attention heatmaps

Checkpoint `checkpoints/demo_english/ckpt_best.pt`. Statistics over 32 held-out windows of 128 tokens; heatmaps for the sentence `'ROMEO:\nO, where art thou? Speak, speak!'`.

![overview](exp9_attention_overview.png)

![selected](exp9_attention_selected.png)

Head labels come from threshold rules (previous-token: >0.5 of attention on token i-1; first-token: >0.5 on position 0; punctuation: >0.3 of mass on punctuation/newline and >2x uniform; broad: normalised entropy >0.75). They are heuristics: check the heatmaps before claiming a head 'does' something.

Heads per label: mixed / local: 5, previous-token: 2, punctuation: 1

| head | prev-token | first-token | punct mass | entropy | label |
|---|---|---|---|---|---|
| L0H0 | 0.97 | 0.01 | 0.13 (uniform 0.10) | 0.00 | previous-token |
| L0H1 | 0.04 | 0.02 | 0.14 (uniform 0.10) | 0.39 | mixed / local |
| L0H2 | 0.56 | 0.01 | 0.15 (uniform 0.10) | 0.36 | previous-token |
| L0H3 | 0.04 | 0.02 | 0.13 (uniform 0.10) | 0.02 | mixed / local |
| L1H0 | 0.11 | 0.02 | 0.15 (uniform 0.10) | 0.59 | mixed / local |
| L1H1 | 0.20 | 0.02 | 0.13 (uniform 0.10) | 0.59 | mixed / local |
| L1H2 | 0.07 | 0.03 | 0.54 (uniform 0.10) | 0.52 | punctuation |
| L1H3 | 0.24 | 0.01 | 0.11 (uniform 0.10) | 0.59 | mixed / local |

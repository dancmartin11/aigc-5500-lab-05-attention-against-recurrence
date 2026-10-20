# AIGC-5500 Lab 5 - Attention Against Recurrence

Attention built from scratch, then tested against recurrence, positional encodings, normalization, and computational cost.

## Prediction

Training on lengths 8, 16, and 24 is expected to improve accuracy at length 32 for both learned and sinusoidal positions compared with training only at length 16.

For learned positions, mixed-length training updates position vectors 0-23 instead of only 0–15, reducing the number of untrained positions used at length 32. However, positions 24–31 remain untrained, so generalization is expected to remain limited.

For sinusoidal positions, exposure to several lengths may encourage a more transferable ordering rule. Improvement is expected, but fixed sinusoidal vectors do not guarantee extrapolation. Accuracy may decrease at length 48 for both encodings.
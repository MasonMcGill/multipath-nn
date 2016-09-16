# Narrative

- Start with a small convnet to predict digit labels
- Add DS routing to short-circuit the network, saving computation
- Compare with CR routing

# "Tricks"
- Use either decision smoothing or cost regression.
- Use spatial pyramids for spatial data.
- Adjust learning rates depending on data flow.
- Use ResNets (maybe).
- Do error mapping (maybe).
- Anneal the routing policy (maybe).

# Experiments

- Static routing vs. dynamic routing (experiment 1)
- Decision smoothing vs. cost regression (experiment 1)
- Trees vs. cascades (experiment 1)
- Optimistic vs. pragmatic cost regression (test informally)
- Generalization error estimation (test informally)
- Spatial pyramids (test informally)
- Subsignal routing (experiment 2)
- Spiking (experiment 3)
- Growing vs. planting (possible follow-up)
- Hybrid convolutional/fully-connected layers (possible follow-up)

# Performance Measurements

- Single Scale, Confident Error — 83-86% accuracy
- Single Scale, Hedging Error — 84% accuracy
- Multicale, Hedging Error — 81% accuracy
- Multicale, Hedging Error, ResNet — 78% accuracy, degrading to 70%
- Single Scale, Confident Error, No LLN — 76% accuracy
- Multicale, Confident Error, Multiscale LLN — 85% accuracy, degrading to 83%
- Multicale, Confident Error, ResNet, Multiscale LLN, Discrete LR Steps — 85.5% accuracy, degrading to 84.5%
- Multicale, Confident Error, Multiscale LLN, Discrete LR Steps — 87.5% accuracy

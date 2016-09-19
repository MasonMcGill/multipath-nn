# Narrative
- Start with a small convnet to predict image labels
- Add DS routing to short-circuit the network, saving computation
- Compare with CR routing

# "Tricks"
- Use either decision smoothing or cost regression.
- Use spatial pyramids for spatial data.
- Adjust learning rates depending on data flow.
- Do error mapping (maybe).

# Experiments
- Static routing vs. dynamic routing (experiment 1)
- Decision smoothing vs. cost regression (experiment 1)
- Trees vs. cascades (experiment 1)
- Optimistic vs. pragmatic cost regression (test informally)
- Error mapping (test informally)
- Subsignal routing (experiment 2)
- Spiking (experiment 3)
- Growing vs. planting (possible follow-up)
- Hybrid convolutional/fully-connected layers (possible follow-up)

# Figures
- CIFAR-10 image
- local-luminance-normalized image
- convnet
- multipath convnet
- multiscale, multipath convnet
- DS routing illustration

# To-Do
- Try error mapping.

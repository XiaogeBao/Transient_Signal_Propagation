# Primate cortex: theoretical and simulated noise responses

Install the dependencies with `python -m pip install -r requirements.txt`. From this directory, open `03_Primate_Cortex.ipynb` in Jupyter and run all cells. The notebook reads the bundled simulation results and writes six PDFs into figures/. To regenerate the simulated data, run `python generate.py` first.

The two groups of panels show time, area and peak for autocovariance and cross-covariance. Each panel compares 29 excitatory populations. Noise is injected into V1 E. The cross-lag sign follows NewNoiseLoad.ipynb.

The anatomical data and supplied model scripts come from the project described in model_source/README.md. The original H_matrix.npy was absent. generate.py constructs H by permuting both axes of the supplied W_matrix.npy from interleaved E/I order into 29 E followed by 29 I. This is the permutation used in the supplied model's eigenmode code. W was independently reconstructed from subgraph_data_macaque.pkl using the supplied connectivity function and matched the supplied W exactly. Exact equality to the historical H cannot be checked.

simulation.py implements the NewNoise Euler simulation and covariance calculations. The run uses T=200000 ms, dt=0.05 ms, one white-noise realization with q00=1 and seed zero, 30% burn-in, 1 ms saved spacing and a 10000 ms lag window. Streaming the sampled trajectory and computing FFTs by region reduce peak memory. results/primate_noise.npz contains H and the summary arrays required for the figure. theory.py contains the theoretical calculations from NewNoiseLoad. The model_source directory preserves the original reference scripts, which are not required to run the figure.

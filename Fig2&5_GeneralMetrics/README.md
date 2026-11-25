# StructuralEffect figures

Install Python dependencies with `python -m pip install -r requirements.txt`. From this folder run `python generate.py --status` to inspect output coverage. Run `python generate.py` to build both datasets, or `python generate.py --panels constant pulse square` for the deterministic inputs. A job runs if any of its outputs are missing. Use `--force` after changing parameters. Original simulation grids and ensemble sizes can require substantial time and memory.

Open **01_Response_Metrics.ipynb** for theory-versus-simulation comparisons (60 arrays in pic_two/) and **02_Directed_Chain_Scaling.ipynb** for directed-chain and ER distance curves (22 arrays in scaling/). Both notebooks read data directly. Graphics are exported under figures/.

Each generator is grouped by input and topology under generators/. Constant and pulse ER scaling pool 100 realizations. Their first ten also supply the comparison panels, preserving each figure’s ensemble size while sharing simulations. This couples the two samples across figures. New unseeded draws can differ from the original results. The constant ER scaling exports retain the original Random_constant.plot_line mapping of its internal peak and amp variables.

## Generate fresh noise results

Run `python generate.py --panels noiseauto noisecov noise-scaling --force` to replace existing noise summaries. Both notebooks consume these arrays. The supplied ZIP includes the noise summaries generated with the defaults. Deterministic panel generation is unchanged.

The noise model uses unit leakage and unit edge weights. Gaussian input drives node zero. The generator filters independent periodic white-noise records through the linear network's sampled impulse response. It averages their cross-spectra to estimate auto- and cross-covariances. Each record has its sample mean removed. Lag integrals use a finite symmetric window; these finite-record estimates can vary with record length, seed and window. Autocorrelation timing uses a fitted exponential and a 1/e crossing. Cross-covariance uses its sampled maximum and the corresponding signed lag. Theoretical estimates use inverse-matrix response moments and midpoint quadrature for the stationary variance. Zero observed lags are excluded from the timing-relative-error denominator.

Edit SEED, DT, N, SAMPLES, RECORDS and TRIALS in generators/fresh_noise.py. Defaults use 100 nodes, 64 independent records of 4096 samples at dt=0.1, and ten stochastic graph realizations per family. The directed chain is used for distance scaling and the bidirectional chain for comparisons. ER realizations and noise records are shared across requested outputs. Reusing the same seed reproduces a given topology and trial regardless of which panels are selected.

The constant-input ER generator uses seed 20261125. Its two comparison files contain the first ten distinct graph realizations (100 nodes per graph), correcting the earlier repetition of the first realization. Its three distance summaries pool all 100 graphs. The 100,000-point Euler recurrence is evaluated in blocks and retains the original time labels and threshold-crossing convention.

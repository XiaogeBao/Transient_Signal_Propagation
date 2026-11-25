# Truncation plots

Install NumPy, Matplotlib and JupyterLab. Open 04_Truncation.ipynb from this directory and run all cells. It exports 31 panels plus the shared path-length legend. Pre-rendered PDFs are included.

data/truncation.npz holds the paired 5-by-3 error and rank matrices printed in the saved outputs of Truncation.ipynb. Rows are truncation orders 1–5, columns are path lengths 1–3. Values retain the displayed precision and NaNs. This is a figure-reproduction package. It does not repeat the upstream network experiments.

Case names follow the input datasets actually selected by the original notebook, including its heterogeneous-weight variants. Distinct export prefixes prevent the two cross_im_m variants from overwriting one another.

In the supplied notebook, the cross_ij_m branch plots peak-response matrices again under the Peak Time label for all three path indices. Those three panels reproduce the saved output. Actual peak-time values for those panels cannot be recovered from their printed matrices.

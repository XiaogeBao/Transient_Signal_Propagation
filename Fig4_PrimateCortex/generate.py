from pathlib import Path
import numpy as np
from Whole.Fig4_PrimateCortex.simulation import run_mc

ROOT = Path(__file__).resolve().parent

def main():
    w = np.load(ROOT / "data/W_matrix.npy")
    order = np.r_[np.arange(0,len(w),2), np.arange(1,len(w),2)]
    h = w[np.ix_(order, order)]
    result = run_mc(h)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "primate_noise.npz", H=h, q00=result["params"]["q00"],
                        **result["metrics_avg"])

if __name__ == "__main__":
    main()

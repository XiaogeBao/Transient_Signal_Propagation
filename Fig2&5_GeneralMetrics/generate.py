"""Build missing response-comparison and distance-scaling data sets."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METRICS = {"constant": ("exptime", "exptimeconstant"), "pulse": ("decaytime", "peaktime", "peak"),
           "square": ("peak",), "noiseauto": ("exptime", "exptimeconstant"),
           "noisecov": ("peak", "peaktime")}
OTHER = ("small-world", "scale-free", "geometric")


def jobs():
    for panel, metrics in METRICS.items():
        noise = panel.startswith("noise")
        chain = "undirectedchain" if noise else "directedchain"
        if noise:
            label = "auto" if panel == "noiseauto" else "cov"
            groups = [(f"noise_{label}_chain", [chain]),
                      (f"other_{label}noise", ["regular", "ER_random" if label == "cov" else "random", *OTHER])]
        else:
            groups = [(panel, [chain] if panel == "pulse" else [chain, "regular"]),
                      ("random_" + panel, ["random"]),
                      ("other_" + panel, ["regular", *OTHER] if panel == "pulse" else OTHER)]
        for script, graphs in groups:
            outputs = [f"pic_two/{panel}_{metric}_{g}_equal.npy" for metric in metrics for g in graphs]
            if not noise and (chain in graphs or "random" in graphs):
                suffix = "_ER" if "random" in graphs else ""
                names = ["peak_response", "amplification"]
                if panel != "square":
                    names.append("time_constant")
                outputs.extend(f"scaling/{panel}_{name}{suffix}.npy" for name in names)
            yield panel, "generators/fresh_noise.py" if noise else f"generators/{panel}/{script}.py", outputs
    yield "noise-scaling", "generators/fresh_noise.py", [
        f"scaling/noise_{name}{suffix}.npy" for name in ("peak_response", "time_constant", "amplification")
        for suffix in ("", "_ER")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", choices=[*METRICS, "noise-scaling"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true", help="List missing output counts without running jobs")
    args = parser.parse_args()
    noise = set()
    noise_files = []
    for panel, script, files in jobs():
        if args.panels and panel not in args.panels:
            continue
        missing = [name for name in files if not (ROOT / name).is_file()]
        print(f"{script}: {len(files)-len(missing)}/{len(files)} outputs present", flush=True)
        if panel in ("noiseauto", "noisecov", "noise-scaling"):
            if missing or args.force:
                noise.add(panel)
                noise_files.extend(files)
            continue
        if args.status or (not missing and not args.force):
            continue
        for folder in ("scaling", "pic_two"):
            (ROOT / folder).mkdir(exist_ok=True)
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=os.pathsep.join([str(ROOT), os.getenv("PYTHONPATH", "")]))
        result = subprocess.run([sys.executable, script], cwd=ROOT, env=env)
        if result.returncode:
            sys.exit(f"{script} failed (exit {result.returncode}). See its traceback above.")
        for name in files:
            if not (ROOT / name).is_file():
                raise FileNotFoundError(name)

    if noise and not args.status:
        for folder in ("scaling", "pic_two"):
            (ROOT / folder).mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        subprocess.run([sys.executable, "generators/fresh_noise.py", *sorted(noise)],
                       cwd=ROOT, env=env, check=True)
        for name in noise_files:
            if not (ROOT / name).is_file():
                raise FileNotFoundError(name)


if __name__ == "__main__":
    main()

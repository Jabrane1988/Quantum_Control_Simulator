import argparse
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(prog="qcsim", description="Quantum Control Systems Simulator")
    p.add_argument("experiment", choices=["rabi", "t1", "ramsey", "drag-opt"])
    args = p.parse_args()

    module_map = {
        "rabi": "qcsim.experiments.rabi",
        "t1": "qcsim.experiments.t1_decay",
        "ramsey": "qcsim.experiments.ramsey",
        "drag-opt": "qcsim.experiments.drag_optimize",
    }

    mod = module_map[args.experiment]
    raise SystemExit(subprocess.call([sys.executable, "-m", mod]))

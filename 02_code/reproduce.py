"""Reproduce in a new directory without overwriting the delivered evidence."""

import argparse, shutil, json
from pathlib import Path
from dnsids.train import run
from dnsids.select import select
from dnsids.evaluate import evaluate_run
from dnsids.cascade import run as cascade


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--with-llm", action="store_true")
    p.add_argument("--endpoint", default="http://127.0.0.1:8765")
    a = p.parse_args()
    original = Path(__file__).resolve().parents[1]
    dest = a.output.resolve()
    if dest.exists():
        raise FileExistsError("Output must be a new directory")
    if original == dest or original in dest.parents:
        raise ValueError("Choose a sibling folder outside the immutable submission")
    dest.mkdir(parents=True)
    for name in ["03_config", "04_data"]:
        shutil.copytree(original / name, dest / name)
    (dest / "06_results").mkdir()
    (dest / "09_reproducibility").mkdir()
    if a.retrain:
        # These copied derived locks must be replaced only in the new reproduction.
        for name in ["selection_lock.json", "cascade_validation_lock.json"]:
            (dest / "03_config" / name).unlink(missing_ok=True)
        for name in ["experiments.json", "refinements.json"]:
            for conf in json.loads((dest / "03_config" / name).read_text()):
                run(dest, conf)
        select(dest)
        for conf in json.loads((dest / "03_config/seed_repeats.json").read_text()):
            run(dest, conf)
    else:
        shutil.copytree(original / "06_results/runs", dest / "06_results/runs")
    selection = json.loads((dest / "03_config/selection_lock.json").read_text())
    if a.with_llm:
        cascade(dest, "validation", a.endpoint)
    for name in selection["evaluate_runs"]:
        evaluate_run(dest, name, ["test", "external_test"])
    if a.with_llm:
        for split in ["test", "external_test"]:
            cascade(dest, split, a.endpoint)
    print("Reproduction saved to", dest)


if __name__ == "__main__":
    main()

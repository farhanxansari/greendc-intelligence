"""Run the full GreenDC analytics pipeline in dependency order.

Run: python -m greendc.pipeline              (core pipeline, a few minutes)
     python -m greendc.pipeline --with-eval  (adds the 30-trial anomaly evaluation)
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ["greendc.data.esif"],                          # raw parquet -> hourly clean data
    ["greendc.models.forecast"],                    # day-ahead forecasting
    ["greendc.models.forecast", "--horizon", "1"],  # hour-ahead forecasting
    ["greendc.models.overhead"],                    # expected-overhead model
    ["greendc.anomaly.detect"],                     # anomaly events (needs overhead)
    ["greendc.pue.metrics"],                        # PUE tables (needs anomaly flags)
    ["greendc.recommend.engine"],                   # recommendations (needs all of the above)
]
EVAL_STEP = ["greendc.anomaly.evaluate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-eval", action="store_true")
    steps = STEPS + ([EVAL_STEP] if parser.parse_args().with_eval else [])

    t_all = time.time()
    for step in steps:
        label = " ".join(step)
        print(f"\n=== {label} ===", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, "-m", *step])
        if result.returncode != 0:
            sys.exit(f"Step failed: {label}")
        print(f"--- done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nPipeline complete in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
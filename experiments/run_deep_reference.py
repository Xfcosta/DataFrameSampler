from __future__ import annotations

from pathlib import Path

from experiments.datasets import DATASET_CONFIGS
from experiments.deep_reference import run_deep_reference_comparison_for_config
from experiments.workflow import load_dataset, resolve_project_root, working_dataframe


def main() -> None:
    root = resolve_project_root(Path(__file__).resolve())
    config = DATASET_CONFIGS["adult"]
    dataframe = working_dataframe(load_dataset(config, root=root), config)
    report = run_deep_reference_comparison_for_config(
        config,
        dataframe,
        results_dir=root / "experiments" / "results",
    )
    print(report)


if __name__ == "__main__":
    main()

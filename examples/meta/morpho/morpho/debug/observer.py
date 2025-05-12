# pyre-unsafe
import logging
import math
import pickle
import traceback
from collections import defaultdict
from traceback import FrameSummary
from typing import Any, Dict, List, Literal, NamedTuple

import torch
from matplotlib import pyplot as plt
from morpho.cli import cli

logger = logging.getLogger(__name__)


class Observation(NamedTuple):
    category: str
    frames: List[FrameSummary]
    value: torch.Tensor


Variant = Literal["reference", "reassociated", "noise", "experiment"]


class Run(NamedTuple):
    run: str
    variant: Variant
    sample: int
    observations: List[Observation]


class Observer:
    def __init__(self, run: str, variant: Variant, sample: int):
        self.run = Run(run, variant, sample, [])

    def observe(self, kind: str, value: torch.Tensor):
        value = value.cpu() if not value.is_cpu else value.clone()
        self.run.observations.append(
            Observation(kind, frames=traceback.extract_stack(), value=value)
        )

    def log(self, step: int, metrics: Dict[str, Any]):
        pass


def compute_differences(data: List[Run]) -> Dict[str, Dict[str, List[float]]]:
    by_run = defaultdict(lambda: defaultdict(list))
    results_by_run = defaultdict(lambda: defaultdict(list))

    for d in data:
        by_category = defaultdict(list)
        for obj in d.observations:
            by_category[obj.category].append(obj.value)

        for category, values in by_category.items():
            plot_name = f"{d.run} {category}"
            by_run[plot_name][d.variant].append(values)

    for run, by_variant in by_run.items():
        references = by_variant["reference"]
        if not references:
            logger.warning(
                f"Skipping {run} because there are no reference runs to compare to."
            )
            continue
        N = len(references[0])
        results_by_variant = results_by_run[run]
        for i in range(N):
            reference_values = torch.stack(tuple(ref[i] for ref in references))
            if (reference_values[:-1] == reference_values[1:]).all():
                reference_mean = reference_values[0]
            else:
                reference_mean = reference_values.mean(0)

            for variant, instances in by_variant.items():
                abs_diff = torch.abs(
                    torch.stack(tuple(inst[i] for inst in instances)) - reference_mean
                )
                results_by_variant[variant].append(abs_diff.mean().item())
    return results_by_run  # type: ignore


def plot_runs(data: List[Run], output_filename: str = "all_runs.png"):
    differences = compute_differences(data)
    plot_differences(differences, output_filename)


def plot_differences(
    differences: Dict[str, Dict[str, List[float]]],
    output_filename: str = "all_runs.png",
):
    variant_colors = {
        "reference": "blue",
        "reassociated": "green",
        "noise": "red",
        "experiment": "purple",
    }
    num_plots = len(differences)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots), squeeze=False)
    for ax, (run_name, variants) in zip(axes.flatten(), differences.items()):
        for variant, data in variants.items():
            assert all(not math.isnan(x) for x in data)
            ax.plot(
                data,
                label=variant,
                marker="o",
                color=variant_colors[variant],
            )
        ax.set_title(f"Mean Absolute Difference for Run: {run_name}")
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Mean Absolute Difference")
        ax.set_ylim(ymin=0)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_filename)


def main(runs: List[str], filename: str = "all_runs.png"):
    return plot_runs([pickle.load(open(run, "rb")) for run in runs], filename)


if __name__ == "__main__":
    cli(main)

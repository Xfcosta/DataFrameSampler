from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProposedSamplerSetup:
    key: str
    label: str
    n_iterations: int
    quantile_guard: float
    max_constraint_retries: int
    calibrate_decoders: bool
    intended_use: str

    @property
    def sampler_config(self) -> dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "quantile_guard": self.quantile_guard,
            "max_constraint_retries": self.max_constraint_retries,
            "calibrate_decoders": self.calibrate_decoders,
        }


PROPOSED_SAMPLER_SETUPS = [
    ProposedSamplerSetup(
        key="fast",
        label="DataFrameSampler fast",
        n_iterations=0,
        quantile_guard=0.1,
        max_constraint_retries=0,
        calibrate_decoders=False,
        intended_use="Smoke tests, quick previews, and cheap notebook checks",
    ),
    ProposedSamplerSetup(
        key="default",
        label="DataFrameSampler default",
        n_iterations=0,
        quantile_guard=0.1,
        max_constraint_retries=5,
        calibrate_decoders=False,
        intended_use="General example-data workflow used by default",
    ),
    ProposedSamplerSetup(
        key="accurate",
        label="DataFrameSampler accurate",
        n_iterations=2,
        quantile_guard=0.1,
        max_constraint_retries=20,
        calibrate_decoders=True,
        intended_use="Slower diagnostic runs where probability quality matters",
    ),
]

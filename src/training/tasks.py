"""Forecast task definitions shared by data adapters, training and CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskName = Literal["state", "first-visible"]
TASK_NAMES: tuple[TaskName, ...] = ("state", "first-visible")


@dataclass(frozen=True)
class ForecastTask:
    """Describe what a prediction horizon means.

    ``state`` is a point forecast for the exact minute ``t + horizon``.
    ``first-visible`` is an onset forecast for the interval ``(t, t + horizon]``.
    Keeping the distinction explicit prevents incomparable labels from being
    grouped under the same horizon name.
    """

    name: TaskName = "state"
    horizon_minutes: int = 1

    def __post_init__(self) -> None:
        if self.name not in TASK_NAMES:
            raise ValueError(f"unknown forecast task: {self.name}")
        if self.horizon_minutes < 1:
            raise ValueError("horizon_minutes must be positive")

    @property
    def is_point_forecast(self) -> bool:
        return self.name == "state"

    @property
    def slug(self) -> str:
        return f"{self.name.replace('-', '_')}_{self.horizon_minutes}m"

    @property
    def description(self) -> str:
        if self.is_point_forecast:
            return (
                "past complete minutes predict degradation state and root service "
                f"at exactly t+{self.horizon_minutes} minute(s)"
            )
        return (
            "past complete minutes predict the first user-visible failure and root service "
            f"within (t, t+{self.horizon_minutes} minute(s)]"
        )

    def apply_to_label_definition(self, definition: dict) -> dict:
        """Return checkpoint metadata with unambiguous task semantics."""
        common = {
            **definition,
            "prediction_horizon_seconds": 60 * self.horizon_minutes,
        }
        if self.is_point_forecast:
            return {
                **common,
                "prediction_task": "failure state at exact future target minute",
                "prediction_window": f"t+{self.horizon_minutes}m",
            }
        return {
            **common,
            "prediction_task": "first user-visible failure within future horizon",
            "prediction_window": f"(t, t+{self.horizon_minutes}m]",
            "failure_reference": "first audited degradation target minute per physical case",
            "post_failure_inputs": "excluded",
            "pre_injection_policy": "healthy control",
        }


__all__ = ["ForecastTask", "TASK_NAMES", "TaskName"]

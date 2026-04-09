"""Curriculum scheduler for progressive attack difficulty."""

from typing import Dict, List, Optional, Tuple


class CurriculumScheduler:
    """Maps training step to active attack families.

    Schedule format: list of (max_step, [families]) tuples.
    Example:
        [(1000, ["P1"]),
         (2000, ["P1", "P2", "P3"]),
         (3000, ["P1", "P2", "P3", "P4", "P5", "P6"])]
    """

    def __init__(self, schedule: List[Tuple[int, List[str]]]):
        # Sort by max_step ascending
        self.schedule = sorted(schedule, key=lambda x: x[0])

    @classmethod
    def from_config(cls, cfg: dict) -> "CurriculumScheduler":
        raw = cfg.get("curriculum", {}).get("schedule", [[1000, ["P1"]]])
        schedule = [(entry[0], entry[1]) for entry in raw]
        return cls(schedule)

    def get_active_families(self, step: int) -> List[str]:
        """Return the list of active attack families at a given training step."""
        for max_step, families in self.schedule:
            if step <= max_step:
                return families
        # Beyond all schedule entries: use the last one
        return self.schedule[-1][1] if self.schedule else ["P1"]

    def get_phase_info(self, step: int) -> Dict:
        """Return current phase info for logging."""
        families = self.get_active_families(step)
        phase_idx = 0
        for i, (max_step, _) in enumerate(self.schedule):
            if step <= max_step:
                phase_idx = i
                break
        else:
            phase_idx = len(self.schedule) - 1

        return {
            "phase": phase_idx + 1,
            "total_phases": len(self.schedule),
            "active_families": families,
            "num_families": len(families),
        }

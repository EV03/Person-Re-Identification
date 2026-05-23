from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot


@dataclass
class PlayerStats:
    player_id: str
    visible_seconds: float = 0.0
    distance_m: float = 0.0
    max_speed_mps: float = 0.0
    sprint_count: int = 0
    heatmap_points: list[tuple[float, float]] = field(default_factory=list)

    @property
    def avg_speed_mps(self) -> float:
        if self.visible_seconds <= 0:
            return 0.0
        return self.distance_m / self.visible_seconds


class PlayerStatsAccumulator:
    """Frame-to-frame player statistic accumulator.

    It expects pitch coordinates in meters. Pixel coordinates should not be fed
    into this class because distance and speed would become meaningless.
    """

    def __init__(self, sprint_threshold_mps: float = 7.0, max_reasonable_speed_mps: float = 12.0) -> None:
        self.sprint_threshold_mps = sprint_threshold_mps
        self.max_reasonable_speed_mps = max_reasonable_speed_mps
        self.stats: dict[str, PlayerStats] = {}
        self._last_position: dict[str, tuple[float, float, float]] = {}

    def update(self, player_id: str, timestamp_sec: float, pitch_x: float, pitch_y: float) -> PlayerStats:
        stat = self.stats.setdefault(player_id, PlayerStats(player_id=player_id))
        stat.heatmap_points.append((float(pitch_x), float(pitch_y)))

        previous = self._last_position.get(player_id)
        if previous is not None:
            prev_time, prev_x, prev_y = previous
            delta_time = max(0.0, float(timestamp_sec) - prev_time)
            if delta_time > 0:
                distance = hypot(float(pitch_x) - prev_x, float(pitch_y) - prev_y)
                speed = distance / delta_time
                if speed <= self.max_reasonable_speed_mps:
                    stat.visible_seconds += delta_time
                    stat.distance_m += distance
                    stat.max_speed_mps = max(stat.max_speed_mps, speed)
                    if speed >= self.sprint_threshold_mps:
                        stat.sprint_count += 1

        self._last_position[player_id] = (float(timestamp_sec), float(pitch_x), float(pitch_y))
        return stat

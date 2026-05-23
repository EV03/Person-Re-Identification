from __future__ import annotations

from app.pipeline.orchestrator import PersonReIdPipeline


class FootballAnalysisPipeline(PersonReIdPipeline):
    """Prepared extension point for the football mode.

    V1 intentionally inherits the stable person ReID pipeline. The next concrete
    steps are to insert team classification after crop creation, ball detection
    after frame read, pitch mapping after detection and stats aggregation after
    coordinates are available.
    """

    pass

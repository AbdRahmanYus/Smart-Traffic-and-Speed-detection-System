"""Shared state for the manual traffic-light override.

"auto" means defer to whatever the traffic light detector sees each
frame. Any other value overrides the detector until it's set back to
"auto". One override is shared across all jobs, which is fine for a
single-operator tool.
"""

import threading

VALID_STATES = {"auto", "red", "amber", "green"}


class TrafficLightOverride:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "auto"

    def get(self) -> str:
        with self._lock:
            return self._state

    def set(self, state: str) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"Unknown traffic light state: {state}")
        with self._lock:
            self._state = state

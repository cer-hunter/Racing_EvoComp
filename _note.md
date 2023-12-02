Changes by Phil:

- `ENGINE_POWER` and `BRAKE_FORCE` changed to 1/5 of original value so things do not happen so fast.

- Remapping of reference frame with respect to the corner itself. (`cx` and `cy`)
    - origin: corner entry point: on the outer side of the straight before corner, just when the track start to turn.
    - `cy` axis: along the straight before corner
        - negative cy means before corner entry
        - we should brake at some negative cy so the car is sufficiently slowed down before corner.
    - `cx` axis: perpendicular to y, indicates lateral position on the straight before corner
    - entry angle: angle with respect to the direction of the straight before the corner
        - 0 means perfectly aligned with the straight (should be the initial condition)
    - exit angle: angle with respect to the direction of the straight after the corner

Further Changes by Hunter:
- updated Action Wrapper to better reflect action space
- updated primitive set
- updated observation space
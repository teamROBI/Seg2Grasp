# Robot module (reference, hardware-specific)

This is the original **UR5e + Robotiq AirPick vacuum gripper** control code used
for the real-robot experiments in the paper, provided **as-is** as a reference
for reproducing the live bin-picking loop. It is **not exercised by the offline
demo** (`scripts/run_demo.py`) and is not required to run the Seg2Grasp pipeline.

- `core/agent.py` — high-level `Agent` (movej/movel, coordinate conversion
  camera→robot, gripper open/close, vacuum feedback).
- `module/` — TCP client, utilities, gripper driver.
- `task/` — depalletization / induction task loops.
- `config.json` — robot IP, home/place/transit joint poses, camera ROI. **Edit
  these for your own cell** (the values are specific to the original setup).

## Dependencies not included
`agent.py` imports `robot_module.module.urx` — a vendored copy of the
[`python-urx`](https://github.com/SintefManufacturing/python-urx) UR driver that
did not survive in the backup. To run this code, install `urx` (`pip install urx`)
and adjust the imports, or drop a `urx/` package under `module/`.

Because this path depends on physical hardware, it is left unmodified rather than
refactored. See `scripts/run_live.py` for a camera + pipeline live loop that runs
without the full robot stack (robot actuation is optional there).

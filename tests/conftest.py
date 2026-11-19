"""Run from tests/ (its pytest.ini keeps pytest away from the pack's own __init__.py): PYTHONPATH=~/AI/ComfyUI pytest"""
import os
import sys

COMFY = os.path.expanduser("~/AI/ComfyUI")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True

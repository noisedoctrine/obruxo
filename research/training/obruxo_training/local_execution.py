"""Keep active local computation awake without editing persistent power settings."""

from contextlib import contextmanager
import ctypes
import os

CONTINUOUS, SYSTEM_REQUIRED = 0x80000000, 0x00000001


def execution_state_api():
    if os.name != "nt":
        return None
    api = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    api.argtypes, api.restype = [ctypes.c_uint32], ctypes.c_uint32
    return api


@contextmanager
def active_computation(*, state_api=None):
    api = state_api if state_api is not None else execution_state_api()
    report = {"requested": False, "released": False, "display_required": False, "persistent_settings_changed": False}
    if api is None:
        yield report
        return
    previous = api(CONTINUOUS | SYSTEM_REQUIRED)
    if not previous:
        raise OSError("Windows refused the scoped active-computation request")
    report["requested"] = True
    try:
        yield report
    finally:
        report["released"] = bool(api(previous | CONTINUOUS))

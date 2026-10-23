"""Bounded Windows publication retries and short-lived JSON snapshot readers."""

import json
import os
from time import monotonic, sleep


def replace_file(source, destination, *, timeout=2.):
    deadline = monotonic() + timeout
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            remaining = deadline - monotonic()
            if getattr(error, "winerror", None) not in (5, 32, 33) or remaining <= 0:
                raise
            sleep(min(.025, remaining))


def read_json_snapshot(path):
    with open(path, encoding="utf-8-sig") as stream:
        text = stream.read()
    return json.loads(text)

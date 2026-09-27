"""What a checkpoint can be named by: a directory, or a Hub repo id (optionally pinned `@revision`).

Kev's `checkpoint.py` keeps the same rule; it lives here too because `inventio systemone --use` has to
validate a name on a machine that has not installed the extra, and importing `checkpoint` would import
torch and transformers to answer a question about a string.
"""

import os
import re

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")


def is_hub_id(run) -> bool:
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None

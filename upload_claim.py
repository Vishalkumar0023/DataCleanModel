"""Atomic temporary-upload handling for the web application."""

import os
from uuid import uuid4


def claim_temp_upload(temp_path):
    """Atomically move a temporary upload into an exclusive processing path.

    Returns the claimed path, or ``None`` when another request has already
    claimed (or completed) the upload.
    """
    folder = os.path.dirname(temp_path)
    claimed_path = os.path.join(folder, f'.processing_{uuid4().hex}.csv')
    try:
        os.replace(temp_path, claimed_path)
    except FileNotFoundError:
        return None
    return claimed_path

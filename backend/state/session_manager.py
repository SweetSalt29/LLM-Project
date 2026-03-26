sessions = {}


def set_active_file(user_id: int, state: dict):
    """
    Set the full session state for a user.
    Expected keys: files (list), types (list), status (str)
    """
    sessions[user_id] = state


def get_active_file(user_id: int):
    """
    Get the current session state for a user.
    Returns None if no files uploaded yet.
    """
    return sessions.get(user_id)


def update_active_file_status(user_id: int, status: str):
    """
    Update only the status field of an existing session.
    Used by background ingestion to mark 'ready' or 'failed: ...'
    """
    if user_id in sessions:
        sessions[user_id]["status"] = status
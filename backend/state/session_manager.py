sessions = {}

def set_active_file(user_id: int, file_name: str, file_type: str):
    sessions[user_id] = {
        "file_name": file_name,
        "file_type": file_type
    }

def get_active_file(user_id: int):
    return sessions.get(user_id)
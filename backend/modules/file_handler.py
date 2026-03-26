import os

UPLOAD_DIR = "data/uploads"


def save_files(files, user_id: int):
    """
    Save multiple uploaded files for a user.
    Accepts files as a list of (filename, bytes) tuples.
    """
    user_dir = os.path.join(UPLOAD_DIR, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    saved_paths = []

    for filename, data in files:
        file_path = os.path.join(user_dir, filename)

        with open(file_path, "wb") as f:
            f.write(data)

        saved_paths.append(file_path)

    return saved_paths
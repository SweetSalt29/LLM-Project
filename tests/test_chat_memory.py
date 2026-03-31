from backend.modules.chat_memory import create_session, save_message, get_session_messages


def test_chat_memory():
    session_id = create_session(user_id=1, mode="rag")

    save_message(session_id, "user", "Hello")
    save_message(session_id, "assistant", "Hi")

    messages = get_session_messages(session_id)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
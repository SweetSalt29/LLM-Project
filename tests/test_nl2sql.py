from backend.modules.nl2sql import is_safe_sql


def test_safe_sql():
    safe, _ = is_safe_sql("SELECT * FROM users")
    assert safe is True


def test_unsafe_sql():
    safe, reason = is_safe_sql("DROP TABLE users")
    assert safe is False
    assert "SELECT" in reason
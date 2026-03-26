def is_structured(file_type):
    return file_type in ["csv", "xlsx", "db", "sqlite"]


def classify_intent(query: str):
    query = query.lower()
    if any(word in query for word in ["sum", "avg", "count", "total"]):
        return "sql"
    return "rag"


def validate(file_type, intent):
    if is_structured(file_type):
        return "nl2sql"

    if intent == "sql":
        return "rag_with_warning"

    return intent


def route_query(file_type: str, query: str):
    if is_structured(file_type):
        return "nl2sql"

    intent = classify_intent(query)
    return validate(file_type, intent)
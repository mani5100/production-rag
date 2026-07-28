import httpx

BASE = "http://localhost:8000/api/v1/chat/"


def chat(message: str, session_id: str | None = None) -> dict:
    payload = {
        "message": message,
        "session_id": str(session_id) if session_id else None,
        "retrieval_filters": None,
    }
    response = httpx.post(BASE, json=payload)
    response.raise_for_status()
    return response.json()


def run():
    print("\n" + "="*60)
    print("MEAL SUBSCRIPTION FLOW TEST")
    print("="*60)

    # Turn 1 — trigger subscription
    print("\n[Turn 1] User: I want to subscribe to meals")
    res = chat("I want to subscribe to meals")
    session_id = res["session_id"]
    print(f"Bot    : {res['answer']}")
    print(f"Session: {session_id}")

    # Turn 2 — meal preference
    user_input = input("\n[Turn 2] Your choice (e.g. 'i want both'): ")
    res = chat(user_input, session_id)
    print(f"Bot    : {res['answer']}")

    # Turn 3 — name
    user_input = input("\n[Turn 3] Your full name: ")
    res = chat(user_input, session_id)
    print(f"Bot    : {res['answer']}")

    # Turn 4 — employee ID
    user_input = input("\n[Turn 4] Your employee ID: ")
    res = chat(user_input, session_id)
    print(f"Bot    : {res['answer']}")

    # Turn 5 — check status
    input("\n[Press Enter when the department has replied to check status]")
    res = chat("What is the status of my meal subscription?", session_id)
    print(f"Bot    : {res['answer']}")

    # Turn 6 — acknowledge
    user_input = input("\n[Turn 6] Send acknowledgment? (yes/no): ")
    res = chat(user_input, session_id)
    print(f"Bot    : {res['answer']}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run()
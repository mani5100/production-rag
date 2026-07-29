from nxb_chatbot.rag.services import llm


def main() -> None:
    response = llm.invoke(
        "Reply with exactly: Groq connection successful"
    )

    print("Content:", response.content)
    print("Metadata:", response.response_metadata)


if __name__ == "__main__":
    main()
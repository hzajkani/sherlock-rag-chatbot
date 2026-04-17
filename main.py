"""CLI entry point for the Sherlock Holmes RAG chatbot."""

from src.cli import main_chat_loop


def main() -> None:
    print("--- Sherlock Holmes RAG Chatbot (CLI) ---")
    main_chat_loop()


if __name__ == "__main__":
    main()

import os
from openai import OpenAI


api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. "
        "Set it as an environment variable before running the program."
    )


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


def chat_with_bot(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        top_p=1,
        max_tokens=4096
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("NVIDIA AI Chatbot")
    print("Type 'exit' to stop.")
    print()

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            answer = chat_with_bot(user_input)
            print("Bot:", answer)
            print()
        except Exception as error:
            print("Error:", error)

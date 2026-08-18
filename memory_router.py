import json


class MemoryRouter:

    def __init__(
        self,
        client,
        model_name
    ):

        self.client = client
        self.model_name = model_name

    # ======================================================
    # CLASSIFY USER MESSAGE
    # ======================================================

    def classify(
        self,
        user_message
    ):

        prompt = """
You are a memory extraction system.

Analyze the user's message and find information
that could be useful in future conversations.

Memory types:

1. working
Temporary information for the current conversation.

2. episodic
Events, activities and experiences that happened.

3. semantic
Stable facts, preferences, interests, goals and skills.

4. procedural
Instructions about how the AI should behave.

For each useful memory return:

type
text
importance
confidence

importance: 0.0 to 1.0

confidence: 0.0 to 1.0

Do NOT store:

- greetings
- normal questions
- temporary small talk
- general knowledge
- assistant statements

Return ONLY JSON.

Example:

{
    "memories": [
        {
            "type": "semantic",
            "text": "User prefers Python.",
            "importance": 0.90,
            "confidence": 0.95
        }
    ]
}

If nothing is useful:

{
    "memories": []
}
"""

        try:

            response = (
                self.client
                .chat
                .completions
                .create(

                    model=self.model_name,

                    messages=[
                        {
                            "role":
                                "system",

                            "content":
                                prompt
                        },
                        {
                            "role":
                                "user",

                            "content":
                                user_message
                        }
                    ],

                    temperature=0,

                    max_tokens=700
                )
            )

            result = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            if result.startswith("```"):

                result = (
                    result
                    .replace(
                        "```json",
                        ""
                    )
                    .replace(
                        "```",
                        ""
                    )
                    .strip()
                )

            data = json.loads(
                result
            )

            return data.get(
                "memories",
                []
            )

        except Exception as e:

            print(
                f"\n[Memory Router Error] {e}"
            )

            return []
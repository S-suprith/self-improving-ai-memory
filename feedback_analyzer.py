import json


class FeedbackAnalyzer:

    def __init__(
        self,
        client,
        model_name
    ):

        self.client = client
        self.model_name = model_name

    # ======================================================
    # ANALYZE FEEDBACK
    # ======================================================

    def analyze(
        self,
        user_feedback,
        memories
    ):

        memory_text = ""

        for item in memories:

            memory_text += (
                f"""
ID: {item['id']}
TYPE: {item['memory_type']}
TEXT: {item['text']}
"""
            )

        prompt = f"""
You are a memory feedback analyzer.

The AI previously used these memories:

{memory_text}

The user provided this feedback:

{user_feedback}

Determine which memories were:

- useful
- incorrect
- outdated
- irrelevant

Return ONLY JSON.

Format:

{{
    "feedback_type": "positive",
    "memory_updates": [
        {{
            "id": "memory_id",
            "action": "reinforce"
        }}
    ]
}}

Allowed feedback_type:

positive
negative
correction
neutral

Allowed actions:

reinforce
reduce_confidence
ignore
update
delete

If the user corrected a memory, identify it.

If no memory is affected:

{{
    "feedback_type": "neutral",
    "memory_updates": []
}}
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
                        }
                    ],

                    temperature=0,

                    max_tokens=500
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

            return json.loads(
                result
            )

        except Exception as e:

            print(
                f"\n[Feedback Analyzer Error] {e}"
            )

            return {
                "feedback_type":
                    "neutral",

                "memory_updates":
                    []
            }
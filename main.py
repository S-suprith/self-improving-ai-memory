import os
import json
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

from memory_manager import MemoryManager
from memory_router import MemoryRouter
from memory_learner import MemoryLearner
from feedback_analyzer import FeedbackAnalyzer


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "openai/gpt-oss-20b"


# ==========================================================
# LOAD ENVIRONMENT
# ==========================================================

load_dotenv()

api_key = os.getenv(
    "GROQ_API_KEY"
)

if not api_key:

    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )


# ==========================================================
# GROQ CLIENT
# ==========================================================

client = Groq(
    api_key=api_key
)


# ==========================================================
# MEMORY COMPONENTS
# ==========================================================

memory_manager = MemoryManager()

memory_router = MemoryRouter(
    client,
    MODEL_NAME
)

memory_learner = MemoryLearner(
    memory_manager
)

feedback_analyzer = FeedbackAnalyzer(
    client,
    MODEL_NAME
)


# ==========================================================
# LAST RETRIEVED MEMORIES
# ==========================================================

last_used_memories = []


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are a personal AI assistant with self-improving memory.

You have:

1. Working memory
2. Episodic memory
3. Semantic memory
4. Procedural memory

Use relevant memories when answering.

Important:

- Never invent user memories.
- Prefer recent and high-confidence information.
- If two memories conflict, prefer the newer reliable memory.
- Follow relevant procedural memories.
- Do not mention internal memory mechanisms unless asked.
"""


# ==========================================================
# RETRIEVE MEMORIES
# ==========================================================

def retrieve_memories(
    query
):

    all_memories = []

    for memory_type in [
        "episodic",
        "semantic",
        "procedural"
    ]:

        results = (
            memory_manager
            .search_memory(
                memory_type,
                query,
                n_results=5
            )
        )

        for item in results:

            item[
                "memory_type"
            ] = memory_type

            all_memories.append(
                item
            )

    # ------------------------------------------------------
    # Adaptive ranking
    # ------------------------------------------------------

    ranked = (
        memory_learner
        .rank_memories(
            all_memories
        )
    )

    return ranked[:6]


# ==========================================================
# BUILD MEMORY CONTEXT
# ==========================================================

def build_memory_context(
    memories
):

    if not memories:

        return (
            "No relevant long-term memories."
        )

    context = ""

    for item in memories:

        context += (
            f"""
[{item['memory_type'].upper()}]

Memory:
{item['text']}

Confidence:
{item['metadata'].get('confidence', 0.5)}

Utility:
{item['metadata'].get('utility', 0.5)}

Retrieval Score:
{item.get('retrieval_score', 0)}
"""
        )

    return context


# ==========================================================
# GENERATE RESPONSE
# ==========================================================

def generate_response(
    user_message
):

    global last_used_memories

    # ------------------------------------------------------
    # Retrieve relevant memories
    # ------------------------------------------------------

    memories = retrieve_memories(
        user_message
    )

    # Save for feedback
    last_used_memories = memories

    # ------------------------------------------------------
    # Build context
    # ------------------------------------------------------

    memory_context = (
        build_memory_context(
            memories
        )
    )

    # ------------------------------------------------------
    # Working memory
    # ------------------------------------------------------

    working_memory = (
        memory_manager
        .get_working_memory()
    )

    # ------------------------------------------------------
    # System message
    # ------------------------------------------------------

    system_message = f"""
{SYSTEM_PROMPT}

Relevant long-term memories:

{memory_context}
"""

    # ------------------------------------------------------
    # Messages
    # ------------------------------------------------------

    messages = [

        {
            "role":
                "system",

            "content":
                system_message
        }

    ]

    messages.extend(
        working_memory[-10:]
    )

    messages.append(
        {
            "role":
                "user",

            "content":
                user_message
        }
    )

    # ------------------------------------------------------
    # Groq
    # ------------------------------------------------------

    response = (
        client
        .chat
        .completions
        .create(

            model=MODEL_NAME,

            messages=messages,

            temperature=0.7,

            max_tokens=1000
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ==========================================================
# PROCESS NEW MEMORY
# ==========================================================

def process_new_memory(
    user_message
):

    print(
        "\n[Memory Learner] "
        "Analyzing new information..."
    )

    memories = (
        memory_router
        .classify(
            user_message
        )
    )

    for item in memories:

        memory_type = item.get(
            "type",
            "semantic"
        )

        text = item.get(
            "text",
            ""
        )

        importance = float(
            item.get(
                "importance",
                0.5
            )
        )

        confidence = float(
            item.get(
                "confidence",
                0.5
            )
        )

        if (
            memory_type == "working"
            or
            not text
        ):

            continue

        # --------------------------------------------------
        # Search existing memories
        # --------------------------------------------------

        existing = (
            memory_manager
            .search_memory(
                memory_type,
                text,
                n_results=1
            )
        )

        # --------------------------------------------------
        # Create new memory
        # --------------------------------------------------

        if not existing:

            if importance >= 0.30:

                memory_manager.add_memory(
                    memory_type,
                    text,
                    importance,
                    confidence
                )

            continue

        candidate = existing[0]

        similarity = (
            candidate["similarity"]
        )

        # --------------------------------------------------
        # Similar memory
        # --------------------------------------------------

        if similarity >= 0.90:

            print(
                "\n[Memory] "
                "Similar memory found."
            )

            memory_id = (
                candidate["id"]
            )

            metadata = dict(
                candidate["metadata"]
            )

            # Reinforce because information
            # is consistent.

            metadata["access_count"] = (
                int(
                    metadata.get(
                        "access_count",
                        0
                    )
                ) + 1
            )

            metadata["success_count"] = (
                int(
                    metadata.get(
                        "success_count",
                        0
                    )
                ) + 1
            )

            metadata["confidence"] = min(
                1.0,
                max(
                    confidence,
                    float(
                        metadata.get(
                            "confidence",
                            0.5
                        )
                    ) + 0.02
                )
            )

            metadata["utility"] = (
                memory_learner
                .calculate_utility(
                    metadata
                )
            )

            metadata["last_accessed"] = (
                datetime.now().isoformat()
            )

            memory_manager.update_metadata(
                memory_type,
                memory_id,
                metadata
            )

            continue

        # --------------------------------------------------
        # Moderately similar information
        # --------------------------------------------------

        if similarity >= 0.70:

            print(
                "\n[Memory] "
                "Related memory found."
            )

            # Keep both for now.
            # Level 6 can learn from feedback.

            if importance >= 0.60:

                memory_manager.add_memory(
                    memory_type,
                    text,
                    importance,
                    confidence
                )

            continue

        # --------------------------------------------------
        # Completely new information
        # --------------------------------------------------

        if importance >= 0.30:

            memory_manager.add_memory(
                memory_type,
                text,
                importance,
                confidence
            )


# ==========================================================
# PROCESS FEEDBACK
# ==========================================================

def process_feedback(
    feedback_text
):

    global last_used_memories

    if not last_used_memories:

        print(
            "\nNo recent memories to evaluate."
        )

        return

    print(
        "\n[Feedback Analyzer] "
        "Analyzing feedback..."
    )

    result = (
        feedback_analyzer
        .analyze(
            feedback_text,
            last_used_memories
        )
    )

    print(
        f"\nFeedback type: "
        f"{result.get('feedback_type')}"
    )

    updates = result.get(
        "memory_updates",
        []
    )

    for update in updates:

        memory_id = update.get(
            "id"
        )

        action = update.get(
            "action"
        )

        # Find corresponding memory
        selected = None

        for item in last_used_memories:

            if item["id"] == memory_id:

                selected = item
                break

        if selected is None:
            continue

        memory_type = (
            selected["memory_type"]
        )

        # --------------------------------------------------
        # REINFORCE
        # --------------------------------------------------

        if action == "reinforce":

            memory_learner.reinforce(
                memory_type,
                memory_id
            )

        # --------------------------------------------------
        # REDUCE CONFIDENCE
        # --------------------------------------------------

        elif action in [
            "reduce_confidence",
            "update"
        ]:

            memory_learner.penalize(
                memory_type,
                memory_id
            )

        # --------------------------------------------------
        # DELETE
        # --------------------------------------------------

        elif action == "delete":

            memory_manager.delete_memory(
                memory_type,
                memory_id
            )

    # ------------------------------------------------------
    # Save feedback log
    # ------------------------------------------------------

    save_feedback_log(
        feedback_text,
        result
    )


# ==========================================================
# SAVE FEEDBACK LOG
# ==========================================================

def save_feedback_log(
    feedback_text,
    result
):

    file_path = (
        "feedback.json"
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

    except:

        data = []

    data.append(
        {
            "timestamp":
                datetime.now().isoformat(),

            "feedback":
                feedback_text,

            "analysis":
                result
        }
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ==========================================================
# SHOW MEMORIES
# ==========================================================

def show_memories():

    data = (
        memory_manager
        .get_all_memory()
    )

    print("\n")
    print("=" * 75)
    print("                 SELF-IMPROVING MEMORY")
    print("=" * 75)

    for memory_type in [
        "episodic",
        "semantic",
        "procedural"
    ]:

        print(
            f"\n{'=' * 20} "
            f"{memory_type.upper()} "
            f"{'=' * 20}"
        )

        documents = data[
            memory_type
        ].get(
            "documents",
            []
        )

        metadatas = data[
            memory_type
        ].get(
            "metadatas",
            []
        )

        ids = data[
            memory_type
        ].get(
            "ids",
            []
        )

        if not documents:

            print(
                "Empty"
            )

            continue

        for i in range(
            len(documents)
        ):

            metadata = (
                metadatas[i]
            )

            print(
                f"\n{i + 1}. "
                f"{documents[i]}"
            )

            print(
                f"   ID: "
                f"{ids[i]}"
            )

            print(
                f"   Importance: "
                f"{metadata.get('importance')}"
            )

            print(
                f"   Confidence: "
                f"{metadata.get('confidence')}"
            )

            print(
                f"   Utility: "
                f"{metadata.get('utility')}"
            )

            print(
                f"   Access Count: "
                f"{metadata.get('access_count')}"
            )

            print(
                f"   Success Count: "
                f"{metadata.get('success_count')}"
            )

            print(
                f"   Correction Count: "
                f"{metadata.get('correction_count')}"
            )

    print(
        "\n" + "=" * 75
    )


# ==========================================================
# SHOW LAST RETRIEVED
# ==========================================================

def show_last_retrieved():

    if not last_used_memories:

        print(
            "\nNo memories were retrieved yet."
        )

        return

    print("\n")
    print(
        "=" * 70
    )

    print(
        "LAST RETRIEVED MEMORIES"
    )

    print(
        "=" * 70
    )

    for i, item in enumerate(
        last_used_memories,
        start=1
    ):

        print(
            f"\n{i}. {item['text']}"
        )

        print(
            f"   Type: "
            f"{item['memory_type']}"
        )

        print(
            f"   Similarity: "
            f"{item['similarity']:.3f}"
        )

        print(
            f"   Retrieval Score: "
            f"{item['retrieval_score']:.3f}"
        )

        print(
            f"   Utility: "
            f"{item['metadata'].get('utility')}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")
    print("=" * 75)
    print("             SELF-IMPROVING AI MEMORY")
    print("                         LEVEL 6")
    print("=" * 75)

    print(
        """
Commands:

memories
    Show all long-term memories.

retrieved
    Show memories used for the latest answer.

feedback good
    Tell the system the retrieved memory was useful.

feedback bad
    Tell the system the retrieved memory was wrong.

feedback correction <text>
    Correct the memory.

clear
    Clear working memory.

forget
    Delete all long-term memories.

learn
    Run low-value memory cleanup.

exit
    Exit the application.
"""
    )

    while True:

        user_input = input(
            "\nYou: "
        ).strip()

        # ==================================================
        # EXIT
        # ==================================================

        if user_input.lower() == "exit":

            print(
                "\nAI: Goodbye!"
            )

            break

        # ==================================================
        # SHOW MEMORIES
        # ==================================================

        if user_input.lower() == "memories":

            show_memories()

            continue

        # ==================================================
        # SHOW RETRIEVED MEMORIES
        # ==================================================

        if user_input.lower() == "retrieved":

            show_last_retrieved()

            continue

        # ==================================================
        # CLEAR WORKING MEMORY
        # ==================================================

        if user_input.lower() == "clear":

            memory_manager.clear_working_memory()

            continue

        # ==================================================
        # DELETE EVERYTHING
        # ==================================================

        if user_input.lower() == "forget":

            confirmation = input(
                "Delete all long-term memories? "
                "(yes/no): "
            )

            if confirmation.lower() == "yes":

                memory_manager.clear_all_long_term_memory()

            continue

        # ==================================================
        # MEMORY LEARNING / FORGETTING
        # ==================================================

        if user_input.lower() == "learn":

            memory_learner.forget_low_value_memories()

            print(
                "\n[Learning Cycle Completed]"
            )

            continue

        # ==================================================
        # FEEDBACK
        # ==================================================

        if user_input.lower().startswith(
            "feedback "
        ):

            feedback_text = (
                user_input[
                    len("feedback "):
                ]
            )

            process_feedback(
                feedback_text
            )

            continue

        # ==================================================
        # EMPTY INPUT
        # ==================================================

        if not user_input:

            continue

        try:

            # ------------------------------------------------
            # Generate response
            # ------------------------------------------------

            answer = generate_response(
                user_input
            )

            print(
                f"\nAI: {answer}"
            )

            # ------------------------------------------------
            # Working memory
            # ------------------------------------------------

            memory_manager.add_working_memory(
                "user",
                user_input
            )

            memory_manager.add_working_memory(
                "assistant",
                answer
            )

            # ------------------------------------------------
            # Learn from new user information
            # ------------------------------------------------

            process_new_memory(
                user_input
            )

        except Exception as e:

            print(
                f"\n[ERROR] {e}"
            )


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()
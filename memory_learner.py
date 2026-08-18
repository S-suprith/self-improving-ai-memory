import math
from datetime import datetime


class MemoryLearner:

    def __init__(
        self,
        memory_manager
    ):

        self.memory_manager = (
            memory_manager
        )

    # ======================================================
    # CALCULATE UTILITY
    # ======================================================

    def calculate_utility(
        self,
        metadata
    ):

        importance = float(
            metadata.get(
                "importance",
                0.5
            )
        )

        confidence = float(
            metadata.get(
                "confidence",
                0.5
            )
        )

        access_count = int(
            metadata.get(
                "access_count",
                0
            )
        )

        success_count = int(
            metadata.get(
                "success_count",
                0
            )
        )

        correction_count = int(
            metadata.get(
                "correction_count",
                0
            )
        )

        # --------------------------------------------------
        # Retrieval success rate
        # --------------------------------------------------

        if access_count > 0:

            success_rate = (
                success_count
                / access_count
            )

        else:

            success_rate = 0.0

        # --------------------------------------------------
        # Correction rate
        # --------------------------------------------------

        if access_count > 0:

            correction_rate = (
                correction_count
                / access_count
            )

        else:

            correction_rate = 0.0

        # --------------------------------------------------
        # Utility
        # --------------------------------------------------

        utility = (

            0.30 * importance

            + 0.30 * confidence

            + 0.30 * success_rate

            - 0.20 * correction_rate
        )

        return max(
            0.0,
            min(
                1.0,
                utility
            )
        )

    # ======================================================
    # REINFORCE
    # ======================================================

    def reinforce(
        self,
        memory_type,
        memory_id
    ):

        memory = (
            self.memory_manager
            .get_memory_by_id(
                memory_type,
                memory_id
            )
        )

        if memory is None:
            return

        metadata = dict(
            memory["metadata"]
        )

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
            float(
                metadata.get(
                    "confidence",
                    0.5
                )
            ) + 0.05
        )

        metadata["importance"] = min(
            1.0,
            float(
                metadata.get(
                    "importance",
                    0.5
                )
            ) + 0.02
        )

        metadata["last_accessed"] = (
            datetime.now().isoformat()
        )

        metadata["utility"] = (
            self.calculate_utility(
                metadata
            )
        )

        self.memory_manager.update_metadata(
            memory_type,
            memory_id,
            metadata
        )

        print(
            "\n[SELF-IMPROVEMENT]"
        )

        print(
            f"Memory reinforced: "
            f"{memory['text']}"
        )

        print(
            f"New utility: "
            f"{metadata['utility']:.3f}"
        )

    # ======================================================
    # NEGATIVE FEEDBACK
    # ======================================================

    def penalize(
        self,
        memory_type,
        memory_id
    ):

        memory = (
            self.memory_manager
            .get_memory_by_id(
                memory_type,
                memory_id
            )
        )

        if memory is None:
            return

        metadata = dict(
            memory["metadata"]
        )

        metadata["access_count"] = (
            int(
                metadata.get(
                    "access_count",
                    0
                )
            ) + 1
        )

        metadata["correction_count"] = (
            int(
                metadata.get(
                    "correction_count",
                    0
                )
            ) + 1
        )

        metadata["confidence"] = max(
            0.0,
            float(
                metadata.get(
                    "confidence",
                    0.5
                )
            ) - 0.15
        )

        metadata["utility"] = (
            self.calculate_utility(
                metadata
            )
        )

        metadata["last_accessed"] = (
            datetime.now().isoformat()
        )

        self.memory_manager.update_metadata(
            memory_type,
            memory_id,
            metadata
        )

        print(
            "\n[SELF-IMPROVEMENT]"
        )

        print(
            f"Memory penalized: "
            f"{memory['text']}"
        )

        print(
            f"New confidence: "
            f"{metadata['confidence']:.3f}"
        )

        print(
            f"New utility: "
            f"{metadata['utility']:.3f}"
        )

    # ======================================================
    # RETRIEVAL RANKING
    # ======================================================

    def rank_memories(
        self,
        memories
    ):

        ranked = []

        for item in memories:

            metadata = item[
                "metadata"
            ]

            similarity = float(
                item.get(
                    "similarity",
                    0
                )
            )

            importance = float(
                metadata.get(
                    "importance",
                    0.5
                )
            )

            confidence = float(
                metadata.get(
                    "confidence",
                    0.5
                )
            )

            utility = float(
                metadata.get(
                    "utility",
                    0.5
                )
            )

            # --------------------------------------------------
            # Learned retrieval score
            # --------------------------------------------------

            score = (

                0.40 * similarity

                + 0.20 * importance

                + 0.20 * confidence

                + 0.20 * utility
            )

            item[
                "retrieval_score"
            ] = round(
                score,
                4
            )

            ranked.append(
                item
            )

        ranked.sort(
            key=lambda x:
                x["retrieval_score"],
            reverse=True
        )

        return ranked

    # ======================================================
    # FORGET LOW UTILITY MEMORY
    # ======================================================

    def forget_low_value_memories(
        self,
        threshold=0.15
    ):

        for memory_type in [
            "episodic",
            "semantic",
            "procedural"
        ]:

            results = (
                self.memory_manager
                .get_all_memory()[
                    memory_type
                ]
            )

            ids = results.get(
                "ids",
                []
            )

            metadatas = results.get(
                "metadatas",
                []
            )

            for i in range(
                len(ids)
            ):

                utility = float(
                    metadatas[i].get(
                        "utility",
                        0.5
                    )
                )

                confidence = float(
                    metadatas[i].get(
                        "confidence",
                        0.5
                    )
                )

                # Only forget memories that are
                # both low utility and low confidence.

                if (
                    utility < threshold
                    and
                    confidence < 0.30
                ):

                    self.memory_manager.delete_memory(
                        memory_type,
                        ids[i]
                    )

                    print(
                        f"[FORGETTING] "
                        f"Removed low-value memory."
                    )
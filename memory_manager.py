import os
import uuid
from datetime import datetime

# ==========================================================
# HUGGING FACE CACHE
# ==========================================================

os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_CACHE"] = "./hf_cache/hub"

import chromadb
from sentence_transformers import SentenceTransformer


class MemoryManager:

    def __init__(self):

        print("\nInitializing memory system...")

        # --------------------------------------------------
        # Persistent ChromaDB
        # --------------------------------------------------

        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        # --------------------------------------------------
        # Four memory layers
        # --------------------------------------------------

        self.episodic = self.client.get_or_create_collection(
            name="episodic_memory"
        )

        self.semantic = self.client.get_or_create_collection(
            name="semantic_memory"
        )

        self.procedural = self.client.get_or_create_collection(
            name="procedural_memory"
        )

        # --------------------------------------------------
        # Working memory
        # --------------------------------------------------

        self.working_memory = []

        # --------------------------------------------------
        # Embedding model
        # --------------------------------------------------

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Memory system ready.")

    # ======================================================
    # WORKING MEMORY
    # ======================================================

    def add_working_memory(
        self,
        role,
        content
    ):

        self.working_memory.append(
            {
                "role": role,
                "content": content
            }
        )

        # Keep only recent conversation
        self.working_memory = (
            self.working_memory[-12:]
        )

    # ======================================================
    # GET WORKING MEMORY
    # ======================================================

    def get_working_memory(self):

        return self.working_memory

    # ======================================================
    # CLEAR WORKING MEMORY
    # ======================================================

    def clear_working_memory(self):

        self.working_memory = []

        print(
            "\n[Working Memory] Cleared."
        )

    # ======================================================
    # GET COLLECTION
    # ======================================================

    def get_collection(
        self,
        memory_type
    ):

        if memory_type == "episodic":
            return self.episodic

        if memory_type == "semantic":
            return self.semantic

        if memory_type == "procedural":
            return self.procedural

        return None

    # ======================================================
    # ADD LONG-TERM MEMORY
    # ======================================================

    def add_memory(
        self,
        memory_type,
        text,
        importance=0.5,
        confidence=0.5
    ):

        collection = self.get_collection(
            memory_type
        )

        if collection is None:
            return None

        if not text.strip():
            return None

        memory_id = str(
            uuid.uuid4()
        )

        embedding = self.embedding_model.encode(
            text
        ).tolist()

        now = datetime.now().isoformat()

        metadata = {

            "memory_type":
                memory_type,

            "importance":
                float(importance),

            "confidence":
                float(confidence),

            "utility":
                0.50,

            "access_count":
                0,

            "success_count":
                0,

            "correction_count":
                0,

            "created_at":
                now,

            "last_accessed":
                now,

            "status":
                "active"
        }

        collection.add(
            ids=[memory_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        print(
            f"\n[{memory_type.upper()} MEMORY CREATED]"
        )

        print(
            f"Memory: {text}"
        )

        return memory_id

    # ======================================================
    # SEARCH MEMORY
    # ======================================================

    def search_memory(
        self,
        memory_type,
        query,
        n_results=8
    ):

        collection = self.get_collection(
            memory_type
        )

        if collection is None:
            return []

        count = collection.count()

        if count == 0:
            return []

        query_embedding = (
            self.embedding_model
            .encode(query)
            .tolist()
        )

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=min(
                n_results,
                count
            ),

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        ids = results.get(
            "ids",
            [[]]
        )[0]

        output = []

        for i in range(
            len(documents)
        ):

            # Chroma cosine distance:
            # smaller = more similar
            similarity = max(
                0.0,
                min(
                    1.0,
                    1.0 - float(distances[i])
                )
            )

            output.append(
                {
                    "id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "similarity": similarity
                }
            )

        return output

    # ======================================================
    # GET MEMORY BY ID
    # ======================================================

    def get_memory_by_id(
        self,
        memory_type,
        memory_id
    ):

        collection = self.get_collection(
            memory_type
        )

        if collection is None:
            return None

        result = collection.get(
            ids=[memory_id],
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = result.get(
            "documents",
            []
        )

        metadatas = result.get(
            "metadatas",
            []
        )

        if not documents:
            return None

        return {
            "id": memory_id,
            "text": documents[0],
            "metadata": metadatas[0]
        }

    # ======================================================
    # UPDATE MEMORY
    # ======================================================

    def update_memory(
        self,
        memory_type,
        memory_id,
        new_text=None,
        metadata=None
    ):

        collection = self.get_collection(
            memory_type
        )

        if collection is None:
            return

        existing = self.get_memory_by_id(
            memory_type,
            memory_id
        )

        if existing is None:
            return

        if new_text is None:
            new_text = existing["text"]

        if metadata is None:
            metadata = existing["metadata"]

        embedding = (
            self.embedding_model
            .encode(new_text)
            .tolist()
        )

        collection.update(

            ids=[memory_id],

            documents=[new_text],

            embeddings=[embedding],

            metadatas=[metadata]
        )

    # ======================================================
    # UPDATE MEMORY METADATA
    # ======================================================

    def update_metadata(
        self,
        memory_type,
        memory_id,
        metadata
    ):

        collection = self.get_collection(
            memory_type
        )

        if collection is None:
            return

        collection.update(
            ids=[memory_id],
            metadatas=[metadata]
        )

    # ======================================================
    # DELETE MEMORY
    # ======================================================

    def delete_memory(
        self,
        memory_type,
        memory_id
    ):

        collection = self.get_collection(
            memory_type
        )

        if collection is None:
            return

        collection.delete(
            ids=[memory_id]
        )

        print(
            f"\n[{memory_type.upper()} MEMORY DELETED]"
        )

    # ======================================================
    # GET ALL MEMORIES
    # ======================================================

    def get_all_memory(self):

        return {

            "episodic":
                self.episodic.get(
                    include=[
                        "documents",
                        "metadatas"
                    ]
                ),

            "semantic":
                self.semantic.get(
                    include=[
                        "documents",
                        "metadatas"
                    ]
                ),

            "procedural":
                self.procedural.get(
                    include=[
                        "documents",
                        "metadatas"
                    ]
                )
        }

    # ======================================================
    # CLEAR MEMORY TYPE
    # ======================================================

    def clear_memory_type(
        self,
        memory_type
    ):

        collection = self.get_collection(
            memory_type
        )

        if collection is None:
            return

        data = collection.get()

        ids = data.get(
            "ids",
            []
        )

        if ids:
            collection.delete(
                ids=ids
            )

        print(
            f"\n[{memory_type.upper()}] Cleared."
        )

    # ======================================================
    # CLEAR ALL
    # ======================================================

    def clear_all_long_term_memory(self):

        self.clear_memory_type(
            "episodic"
        )

        self.clear_memory_type(
            "semantic"
        )

        self.clear_memory_type(
            "procedural"
        )

    # ======================================================
    # MEMORY COUNTS
    # ======================================================

    def memory_counts(self):

        return {

            "working":
                len(self.working_memory),

            "episodic":
                self.episodic.count(),

            "semantic":
                self.semantic.count(),

            "procedural":
                self.procedural.count()
        }
import os
import argparse
import asyncio
import google.generativeai as genai
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# -----------------------------
# Gemini LLM function (async)
# -----------------------------
async def gemini_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    def _call_sync():
        model = genai.GenerativeModel("gemini-2.0-flash")  # dùng model ổn định
        messages = []

        if system_prompt:
            messages.append({"role": "system", "parts": [system_prompt]})

        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append({"role": role, "parts": [content]})

        messages.append({"role": "user", "parts": [prompt]})

        response = model.generate_content(messages)
        return response.text if response and response.text else ""

    return await asyncio.to_thread(_call_sync)


# -----------------------------
# Gemini Embedding function (async)
# -----------------------------
async def gemini_embed(texts):
    def _call_sync():
        model = "models/embedding-001"
        embeddings = []
        for t in texts:
            r = genai.embed_content(model=model, content=t)
            emb = r.get("embedding") if isinstance(r, dict) else getattr(r, "embedding", None)
            if emb is None:
                raise ValueError("Gemini embedding trả về rỗng!")
            embeddings.append(emb)
        return embeddings

    return await asyncio.to_thread(_call_sync)


embedding_func = EmbeddingFunc(
    embedding_dim=768,   # embedding dimension của Gemini-001
    max_token_size=8192,
    func=gemini_embed
)


# -----------------------------
# Main async
# -----------------------------
async def main(api_key, question):
    # config Gemini
    genai.configure(api_key=api_key)

    config = RAGAnythingConfig(
        working_dir="./RAG/rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=gemini_llm_func,
        embedding_func=embedding_func,
    )

    # ingest SWC
    swc_dir = "./RAG/data/SWC1"
    if os.path.exists(swc_dir):
        for fname in os.listdir(swc_dir):
            path = os.path.join(swc_dir, fname)
            if os.path.isfile(path):
                await rag.process_document_complete(
                    file_path=path,
                    output_dir="./output_swc",
                    parse_method="txt"
                )

    # query
    result = await rag.aquery(question, mode="hybrid")
    print("\n=== Query Result ===\n")
    print(result)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-Anything with Gemini API")
    parser.add_argument("--api_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the RAG system")
    args = parser.parse_args()

    asyncio.run(main(args.api_key, args.question))

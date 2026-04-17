# 🔍 Sherlock Holmes — Canon Chatbot

A production-style, Ragas-evaluated **Retrieval-Augmented Generation (RAG)** chatbot that is a verifiable expert on the Sherlock Holmes canon — specifically *The Adventures of Sherlock Holmes* and *The Memoirs of Sherlock Holmes* by Sir Arthur Conan Doyle.

Built with **LlamaIndex**, **Groq** (Llama 3.3 / Llama 3.1 / Gemma 2), **HuggingFace** embeddings, **Streamlit**, and **Ragas** for evaluation.

> Ask it about Irene Adler, the Red-Headed League, the speckled band, the Reichenbach Falls — or try to catch it out with something that is *not* in the canon. It is instructed to refuse gracefully.

---

## ✨ Features

- **Chat UI built with Streamlit** — persistent conversation, sample questions, typing spinner, and the full transcript rendered with `st.chat_message`.
- **Verifiable answers** — every assistant turn exposes its retrieved source passages in an expander. No black box.
- **Four personas** — *Canon Expert*, *Concise Detective*, *Literary Scholar*, *Dr. Watson's Voice*. Every persona inherits the same "answer strictly from the context" guard-rail.
- **Live configuration sidebar** — swap LLM, change temperature, tune `similarity_top_k`, toggle a cross-encoder reranker, toggle HyDE query rewriting, and reset to defaults in one click.
- **"New Chat" control** — clears the transcript and the chat memory so the model starts fresh.
- **Thumbs-up / thumbs-down feedback** — stored per-turn in session state, ready to wire into an analytics sink.
- **CLI entry point** — `python main.py` drops you into a terminal REPL for quick debugging.
- **Full Ragas evaluation harness** — `python evaluate.py` sweeps chunking strategies and reports Faithfulness, Answer Correctness, Context Precision, and Context Recall to CSV.

---

## 🏗️ Architecture

```
┌──────────────────────┐
│  User question       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     (optional)
│  HyDE query rewrite  │ ─── generates a hypothetical answer
└──────────┬───────────┘     and embeds it as the query
           │
           ▼
┌──────────────────────┐
│  Vector retriever    │     HuggingFace all-MiniLM-L6-v2 embeddings
│  (top-k passages)    │     over LlamaIndex's default vector store
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     (optional)
│  Cross-encoder       │ ─── ms-marco-MiniLM-L-6-v2 rerank
│  reranker            │     keeps only the top-N most relevant
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Groq LLM            │     System prompt (persona) +
│  (llama-3.3-70b)     │     retrieved context + chat memory
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Answer + sources    │
└──────────────────────┘
```

---

## 📁 Project structure

```
sherlock-rag-chatbot/
├── app.py                       # Streamlit UI (main entry point)
├── main.py                      # CLI entry point for quick testing
├── evaluate.py                  # Entry point for the Ragas evaluation
│
├── data/
│   ├── The Adventures of Sherlock Holmes.txt
│   └── The Memoirs of Sherlock Holmes.txt
│
├── src/
│   ├── config.py                # All runtime defaults
│   ├── prompts.py               # System-prompt personas
│   ├── model_loader.py          # LLM + embedding factories
│   ├── indexing.py              # Build / load / persist the vector index
│   ├── chat_engine.py           # Chat engine factory (reranker + HyDE aware)
│   ├── query_rewrite.py         # Custom LLM-based query rewriter
│   └── cli.py                   # CLI REPL driver
│
├── ui/
│   ├── sample_questions.py      # Curated welcome-screen questions
│   └── components.py            # Sidebar, sources expander, feedback buttons
│
├── evaluation/
│   ├── evaluation_config.py
│   ├── evaluation_questions.py  # Sherlock-specific Q / A ground truths
│   ├── evaluation_model_loader.py
│   ├── evaluation_helper_functions.py
│   └── evaluation_engine.py
│
├── environment.yml              # Conda environment (primary)
├── requirements.txt             # Pip fallback (Streamlit Cloud, etc.)
├── .env.example                 # Template for the GROQ_API_KEY
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Clone and install

```bash
git clone https://github.com/hzajkani/sherlock-rag-chatbot.git
cd sherlock-rag-chatbot
conda env create -f environment.yml
conda activate sherlock-rag-chatbot
```

If you are not using conda, `pip install -r requirements.txt` inside a fresh virtualenv works too.

### 2. Configure your API key

Copy `.env.example` to `.env` and paste in a Groq API key from [console.groq.com](https://console.groq.com):

```bash
cp .env.example .env
# then edit .env
```

### 3. Run the chatbot

```bash
streamlit run app.py
```

The first run will build the vector store under `local_storage/vector_store/`; subsequent runs reuse it.

### 4. (Optional) Use the CLI instead

```bash
python main.py
```

---

## 🧪 Running the evaluation

The Ragas harness drives the chatbot with the ground-truth questions in [evaluation/evaluation_questions.py](evaluation/evaluation_questions.py) and measures four metrics per answer:

| Metric              | What it measures |
|---------------------|-----------------|
| **Faithfulness**    | Does the answer stick to the retrieved context? |
| **Answer Correctness** | Does the answer match the ground truth semantically? |
| **Context Precision** | Is the retrieved context on-topic? |
| **Context Recall**  | Does the retrieved context cover the ground truth? |

Run it with:

```bash
python evaluate.py
```

Results are written to `evaluation/evaluation_results/` as timestamped CSV files (one detailed per-question file and one averaged summary file per experiment).

> The harness is rate-limited by default because Groq's free tier is aggressive; switch to `evaluate_without_rate_limit` inside [evaluation/evaluation_engine.py](evaluation/evaluation_engine.py) if you are using a paid tier or a local model.

---

## 🔧 Configuration reference

All defaults live in [src/config.py](src/config.py). The Streamlit sidebar exposes the most useful ones at runtime.

| Setting                   | Default                                    | Purpose |
|---------------------------|--------------------------------------------|---------|
| `LLM_MODEL`               | `llama-3.3-70b-versatile`                  | Default Groq model. |
| `AVAILABLE_LLM_MODELS`    | Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B     | Models shown in the UI selector. |
| `LLM_TEMPERATURE`         | `0.1`                                      | Lower = more deterministic answers. |
| `EMBEDDING_MODEL_NAME`    | `sentence-transformers/all-MiniLM-L6-v2`   | Fast, high-quality sentence encoder. |
| `SIMILARITY_TOP_K`        | `4`                                        | Passages retrieved per question. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50`                          | Sentence-splitter configuration. |
| `RERANKER_MODEL_NAME`     | `cross-encoder/ms-marco-MiniLM-L-6-v2`     | Cross-encoder used when reranking is enabled. |
| `RERANKER_TOP_N`          | `3`                                        | Passages kept after the reranker. |
| `USE_HYDE_DEFAULT`        | `False`                                    | Start with HyDE off. |
| `CHAT_MEMORY_TOKEN_LIMIT` | `3900`                                     | Rolling chat memory window. |

---

## 🗺️ Roadmap — where to take this next

This project is a foundation. Two natural next tracks are outlined in the class "What's Next?" brief:

**Path 1 — AI Engineer (deeper optimisation)**
- Sweep alternative Groq / OpenAI / local LLMs and compare Faithfulness vs. Answer Correctness.
- Swap embedding models using the [Hugging Face MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and re-run the evaluation.
- Compare the built-in `HyDEQueryTransform` against the custom rewriter in [src/query_rewrite.py](src/query_rewrite.py).
- Build an LLM-based reranker (prompt the LLM to score 1–10 per passage) and benchmark it against the cross-encoder.

**Path 2 — Product Builder (already delivered here)**
- ✅ Streamlit chat UI with sources, "New Chat", thumbs-up/down, and a full configuration sidebar.
- Next: wire the feedback buttons to an analytics backend; add authentication; ship to Streamlit Community Cloud.

---

## 📚 Acknowledgements

- **Sir Arthur Conan Doyle** — for writing the stories.
- **Project Gutenberg** — for the public-domain text used as source data (eBooks [#1661](https://www.gutenberg.org/ebooks/1661) and [#834](https://www.gutenberg.org/ebooks/834)).
- **LlamaIndex**, **Groq**, **HuggingFace**, **Streamlit**, and **Ragas** for the open-source tooling that makes this possible.

---

## 📄 License

Project code: MIT (feel free to fork and adapt). Source texts: public domain via Project Gutenberg.

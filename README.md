# HealMate RAG Chatbot 🏥
**FAISS + Sentence Transformers + Claude (Anthropic)**

This system upgrades your intent-based chatbot to a full **Retrieval-Augmented Generation (RAG)** pipeline — enabling smarter, context-aware medical responses.

---

## Architecture

```
User Query
    │
    ▼
[Sentence Transformer]  ←── Embeds query into vector
    │
    ▼
[FAISS Index]           ←── Finds top-K similar intent chunks
    │
    ▼
[Relevance Check]       ←── Score threshold filtering
    │
    ├── Low score  →  "Please consult a doctor"
    ├── Direct tag →  Return canned response (greetings etc.)
    └── Medical    →  Pass context to Claude
                            │
                            ▼
                     [Claude Sonnet]  →  Final response
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
# Linux / macOS
export ANTHROPIC_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key-here"
```
Get your API key at: https://console.anthropic.com/

### 3. Place your dataset
Make sure `intents.json` is in the same folder as the scripts.

---

## Usage

### Step 1 — Build the FAISS index (run once)
```bash
python ingest.py
```
This will:
- Load `intents.json`
- Embed all intents using `all-MiniLM-L6-v2`
- Save `healmate_faiss.index` and `healmate_meta.pkl`

### Step 2 — Run the chatbot
```bash
python chatbot.py
```

---

## Example Interaction

```
You: I have a high fever since this morning

HealMate: Based on your symptoms, a high fever can indicate an
infection. Please monitor your temperature closely. If it exceeds
103°F, seek immediate medical attention. Stay hydrated and rest.
You can take Paracetamol (500mg) to help reduce the fever...
```

---

## Files

| File | Purpose |
|------|---------|
| `intents.json` | Your original medical dataset |
| `ingest.py` | Builds FAISS index from intents.json |
| `chatbot.py` | Main RAG chatbot (retrieval + Claude) |
| `requirements.txt` | Python dependencies |
| `healmate_faiss.index` | Generated FAISS vector index |
| `healmate_meta.pkl` | Generated chunk metadata |

---

## Key Parameters (in chatbot.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | 3 | Number of chunks retrieved per query |
| `SIM_THRESHOLD` | 0.35 | Minimum similarity score to use RAG |
| `MAX_TOKENS` | 512 | Max tokens in Claude's response |

---

## Why This Is Better Than Your Original Chatbot

| Feature | Old (Intent-based) | New (RAG) |
|--------|-------------------|-----------|
| Matching | Exact keyword match | Semantic similarity |
| Flexibility | Rigid patterns | Understands paraphrases |
| Responses | Fixed canned replies | Dynamic, contextual answers |
| Scalability | Gets slower with more intents | Fast FAISS search |
| Intelligence | Rule-based | LLM-powered |

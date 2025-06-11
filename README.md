# 🧠 RAGmate - Local RAG for JetBrains AI Assistant

**RAGmate** is an open-source, lightweight server that extends **JetBrains AI Assistant** with actual knowledge of your project.  
It indexes your codebase locally and injects relevant context into prompts — so you get better, context-aware answers without changing your IDE workflow.

⚡ Works with **OpenAI**, **Ollama**, **LM Studio**, and any LLM with an API.  
🔒 No cloud, no lock-in — everything runs **locally**.

## Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/ragmate/ragmate/main/assets/docs/product-demo.gif" width="100%" />
</p>

> JetBrains AI Assistant can't answer: “What patterns does this project use for the file scan?”  
> With RAGmate, it gives a detailed answer — based on real code context.

---

## 🚀 Why RAGmate?

JetBrains AI Assistant is helpful — but lacks real project awareness.  
**RAGmate adds missing context**, without plugins or cloud syncing.

- ✅ Built for JetBrains IDEs
- 🧠 Brings RAG to your local machine
- 🧩 Works with any LLM API (OpenAI, local models, etc)
- 🧼 No framework complexity (no LangChain / LlamaIndex)
- 📁 Local embeddings + semantic search over your codebase

---

## ⚙️ How it works

1. **Index** your project (automatically detects files to scan)
2. **Start** RAGmate server
3. **Connect** your JetBrains IDE to the RAGmate HTTP bridge
4. Ask AI Assistant anything — with real code context

---

## 🛠️ Supported:

- ✅ JetBrains IDEs (via HTTP bridge)
- ✅ Any LLM with a simple `POST /completion` interface
- ✅ Local embeddings (OpenAI, HuggingFace, more coming)

---

## 🧪 Use cases

Ask your AI Assistant:

- “Where is `verify_token()` used?”
- “Explain the login flow in this codebase”
- “How does the error handler work across services?”

RAGmate ensures answers are grounded in your real code.

---

## ✨ Features

- 🧠 Context-aware completions using your project’s actual codebase.
- ⚙️ Integration with JetBrains IDEs via AI Assistant.
- 🔄 Real-time file change tracking and automatic reindexing.
- 🔌 Use any external or local LLM and embedding model.
- 🛡️ Fully local — your code never leaves your machine.

---

## 👤 Who is this for?

- Developers already using JetBrains AI Assistant
- Engineers working with large or legacy codebases
- Teams needing privacy-focused, local AI tools
- Anyone frustrated by AI that “doesn't know the code”

---

## 🚀 Getting Started

### ✅ Prerequisites

- [Docker Compose](https://docs.docker.com/compose/install/)

---

## 🛠️ Installation & Setup

### 1. Add a Docker Compose service

In your project root, create or edit `compose.yml`:

```yaml
services:
  ragmate:
    image: ghcr.io/ragmate/ragmate:latest
    ports:
      - "11434:11434"
    env_file:
      - ./.ragmate.env
    volumes:
      - .:/project
      - ./docker_data/ragmate:/apps/cache
```

> 💡 `./docker_data/ragmate:/apps/cache` — path to the local cache, which can be adjusted as needed.

---

### 2. Add environment variables

Create the `.ragmate.env` file at the project root and add it to `.gitignore`.

#### Required variables:

| Variable               | Description                                |
|------------------------|--------------------------------------------|
| `LLM_MODEL`         | LLM model for generation (e.g., `o3-mini`) |
| `LLM_PROVIDER`| LLM provider (e.g., `openai`, `mistralai`) |
| `LLM_API_KEY`       | Your LLM API key                           |

_The full list of available providers and their models you can find below_

#### Optional variables:

| Variable                | Description                                                                                     | Default value                                                                         |
|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `REINDEX_AFTER_N_CHANGES`| After how many file changes to rebuild the index                                                | `50`                                                                                  |
| `FRAMEWORK`             | Specify the framework used for more accurate answers (e.g., `django`, `spring`, `nextjs`, etc.) | — (not set)                                                                           |
| `TEXT_FILE_EXTENSIONS`  | File extensions to index (square brackets, comma-separated, without spaces, and in quotes)      | `[".py", ".js", ".ts", ".php", ".java", ".rb", ".go", ".cs", ".rs", ".html", ".css"]` |
| `LLM_EMBEDDING_MODEL`  | Embedding model (e.g., `text-embedding-3-large`, etc.)                                          | `microsoft/codebert-base`                                                             |
| `EMBEDDING_API_KEY`  | Embedding model API key                                                                         | - (not set)                                                                           |
| `EMBEDDING_PROVIDER`  | Embedding provider (e.g., `openai`, `huggingface`)                                                  | `huggingface`                                                                         |
| `LLM_BASE_URL`  | Base URL for LLM API (only if using a proxy)                                                    | - (not set)                                                                           |
| `LLM_TEMPERATURE`  | Parameter that controls the randomness of the model's output                                    | `0.7`                                                                                 |

> 🧾 **File Ignoring**:
> Ragmate automatically excludes files and folders specified in `.gitignore` and `.aiignore` located in the project root.

> Example of `.ragmate.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=o3-mini
LLM_API_KEY=sk-...
FRAMEWORK=django
TEXT_FILE_EXTENSIONS=[".py",".html",".css"]
```

#### LLM providers and their models

| `LLM_PROVIDER`               | Models                                                               |
|------------------------|----------------------------------------------------------------------|
| `openai`         | [OpenAI docs](https://platform.openai.com/docs/models)               |
| `anthropic`| [Anthropic docs](https://docs.anthropic.com/en/docs/models-overview) |
| `google-genai`       | [Gemini API docs](https://ai.google.dev/gemini-api/docs/models)      |
| `mistralai`       | [Mistral AI docs](https://docs.mistral.ai/getting-started/models/)   |
| `xai`       | [xAI docs](https://docs.x.ai/docs/models#models-and-pricing)         |
| `deepseek`       | [DeepSeek docs](https://api-docs.deepseek.com/quick_start/pricing)   |

#### Embedding models

| `EMBEDDING_PROVIDER`               | Models                                                               |
|------------------------|----------------------------------------------------------------------|
| `openai`         | [OpenAI docs](https://platform.openai.com/docs/models)               |
| `huggingface`| [HuggingFace models](https://huggingface.co/models?other=embeddings) |

---

### 3. Run the container

```bash
docker compose up -d
```

---

### 4. Configure JetBrains AI Assistant

1. Open **Settings** → **Tools** → **AI Assistant** → **Models**
2. Enable **Enable Ollama**
3. Click **Test Connect**
4. In **Core features** and **Instant helpers**, select `ollama/ragmate`

---

## Demo Setup

<p align="center">
  <img src="https://raw.githubusercontent.com/ragmate/ragmate/main/assets/docs/demo-setup.gif" width="100%" />
</p>

---

## 📄 License

Licensed under the [Apache 2.0 License](LICENSE).

---

## 🤝 Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## ⭐️ Like the idea?

Star the repo and share feedback — we’re building in the open.

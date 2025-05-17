# Ragmate

**Local RAG server for code editors (JetBrains supported).**
Scans your codebase, builds a local context index, and connects to any external LLM for context-aware code generation.

---

## Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/ragmate/ragmate/main/assets/docs/product-demo.gif" width="100%" />
</p>

The demo shows how Ragmate extends JetBrains AI Assistant with a RAG context.

The prompt that was used for that:
```
What patterns does this project use for the files scan?
```

The comparison between the default JetBrains AI Assistant with the GPT 4.1 mini LLM model and Ragmate with GPT 4.1 mini and local RAG.

Ragmate environment variables that were used in this demo:
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=sk-...
TEXT_FILE_EXTENSIONS=[".py"]
FRAMEWORK=fastapi
EMBEDDING_PROVIDER=gpt4all
LLM_EMBEDDING_MODEL=nomic-embed-text-v1.5.f16.gguf
```

---

## ✨ Features

- 🧠 Context-aware completions using your project’s actual codebase.
- ⚙️ Integration with JetBrains IDEs via AI Assistant.
- 🔄 Real-time file change tracking and automatic reindexing.
- 🔌 Use any OpenAI-compatible LLM and embedding model.
- 🛡️ Fully local — your code never leaves your machine.

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
| `LLM_PROVIDER`| Embedding model (e.g., `openai`)           |
| `LLM_API_KEY`       | Your LLM API key                           |

#### Optional variables:

| Variable                | Description                                                                                     | Default value                                                               |
|-------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `REINDEX_AFTER_N_CHANGES`| After how many file changes to rebuild the index                                                | `100`                                                                       |
| `FRAMEWORK`             | Specify the framework used for more accurate answers (e.g., `django`, `spring`, `nextjs`, etc.) | — (not set)                                                                 |
| `TEXT_FILE_EXTENSIONS`  | File extensions to index (square brackets, comma-separated, without spaces, and in quotes)      | `[".py",".js",".ts",".php",".java",".rb",".go",".cs",".rs",".html",".css"]` |
| `LLM_EMBEDDING_MODEL`  | Embedding model (e.g., `text-embedding-3-large`, etc.)                                          | `all‑MiniLM‑L6‑v2.gguf2.f16.gguf`                                           |
| `EMBEDDING_API_KEY`  | Embedding model API key                                                                         | - (not set)                                                                 |
| `EMBEDDING_PROVIDER`  | Embedding provider (e.g., `openai`, `gpt4all`)                              | `gpt4all` |

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

| `EMBEDDING_PROVIDER`               | Models                                                             |
|------------------------|--------------------------------------------------------------------|
| `openai`         | [OpenAI docs](https://platform.openai.com/docs/models)             |
| `gpt4all`| [GPT4All docs](https://docs.gpt4all.io/old/gpt4all_python_embedding.html#supported-embedding-models) |

_Note: `gpt4all` runs the embedding model locally and it does not require `LLM_EMBEDDING_MODEL`_

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

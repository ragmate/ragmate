# Ragmate

**Local RAG server for code editors (JetBrains supported).**
Scans your codebase, builds a local context index, and connects to any external LLM for context-aware code generation.

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

| Variable               | Description                                             |
|------------------------|---------------------------------------------------------|
| `LLM_MODEL`         | OpenAI LLM model for generation (e.g., `o3-mini`)       |
| `LLM_EMBEDDING_MODEL`| OpenAI embedding model (e.g., `text-embedding-3-large`) |
| `LLM_API_KEY`       | Your OpenAI API key                                     |

#### Optional variables:

| Variable                | Description                                                                                     | Default value                                         |
|-------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `REINDEX_AFTER_N_CHANGES`| After how many file changes to rebuild the index                                                | `100`                                                 |
| `FRAMEWORK`             | Specify the framework used for more accurate answers (e.g., `django`, `spring`, `nextjs`, etc.) | — (not set)                                           |
| `TEXT_FILE_EXTENSIONS`  | File extensions to index (square brackets, comma-separated, without spaces, and in quotes)                           | `[".py",".js",".ts",".php",".java",".rb",".go",".cs",".rs",".html",".css"]` |

> 🧾 **File Ignoring**:
> Ragmate automatically excludes files and folders specified in `.gitignore` and `.aiignore` located in the project root.

> Example of `.ragmate.env`:

```env
LLM_MODEL=o3-mini
LLM_EMBEDDING_MODEL=text-embedding-3-large
LLM_API_KEY=sk-...
REINDEX_AFTER_N_CHANGES=100
FRAMEWORK=django
TEXT_FILE_EXTENSIONS=[".py",".html",".css"]
```

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

## Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/ragmate/ragmate/main/assets/docs/demo.gif" width="100%" />
</p>

---

## 📝 Future Plans

VS Code Editor Support: We plan to integrate Ragmate with Visual Studio Code to support this popular IDE.

New LLM Integrations: Support for additional language models is coming soon, including:

* Gemini
* Claude
* Mistral
* DeepSeek

---

## 📄 License

Licensed under the [Apache 2.0 License](LICENSE).

---

## 🤝 Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details (coming soon).

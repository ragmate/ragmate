# Contributing to Ragmate

Thanks for your interest in contributing to Ragmate! 🚀
We welcome contributions of all kinds — code, documentation, bug reports, ideas, and feedback.

---

## 📌 How to Contribute

### 1. Report Issues
If you've found a bug or have a feature request, please [open an issue](https://github.com/ragmate/ragmate/issues).  
Be as descriptive as possible: what happened, what you expected, and steps to reproduce.

---

### 2. Suggest Features
We’re actively improving Ragmate. If you have ideas, open a new issue and use the `feature` label.

---

### 3. Contribute Code

#### 🛠 Prerequisites
- Docker + Docker Compose
- Python 3.10+
- Git

#### 📥 Clone the repo
```bash
git clone https://github.com/ragmate/ragmate.git
cd ragmate
```

#### 🧪 Run the project locally
1. Create a virtual environment
```bash
python -m venv .venv
```

2. Install dependencies
```bash
pip install -r requirements/local.txt
```

3. Run the app
```bash
uvicorn app.main:app --reload --reload-dir app --port 11434
```

#### 🔄 Set up pre-commit
We use `pre-commit` to ensure consistent formatting and static analysis.

Install and set up hooks:
```bash
pip install pre-commit
pre-commit install
```

To run checks manually:
```bash
pre-commit run --all-files
```

---

### ✍️ Commit Style

Please follow the [Conventional Commits](https://www.conventionalcommits.org/) convention:

```
<type>: short description
```

Examples:
- `feat: add support for Gemini model`
- `fix: handle error when env file is missing`
- `docs: update README with setup instructions`

**Allowed types:**
- `feat` – a new feature is introduced
- `fix` – a bug fix
- `docs` – documentation updates (e.g. README)
- `style` – formatting only (white-space, commas, etc.)
- `refactor` – code changes that don’t fix bugs or add features
- `perf` – performance improvements
- `test` – adding or fixing tests
- `chore` – other changes (e.g. dependency updates)
- `ci` – CI configuration changes
- `build` – build system or dependency-related changes
- `revert` – reverts a previous commit

---

### ✅ Submit a Pull Request
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and commit (`git commit -m "feat: add new feature"`)
4. Push to your fork and open a Pull Request

Please ensure all pre-commit hooks pass before submitting.

---

## 🤝 Code of Conduct
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) when interacting in this project.

---

## 🙏 Thank You!
Your contributions help make Ragmate better for everyone. We’re excited to have you here!

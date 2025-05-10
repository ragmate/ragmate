run:
	@uvicorn app.main:app --reload --reload-dir app --port 11434

up:
	@docker compose up -d
	@docker images -q -f dangling=true | xargs docker rmi -f

down:
	@docker compose down

build:
	@docker build -t ragmate/ragmate:latest -f Dockerfile .

linters:
	@pre-commit run --all-files -c .pre-commit-config.yaml

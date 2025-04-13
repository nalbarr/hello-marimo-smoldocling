help:
	@echo make lint
	@echo make format
	@echo make run
	@echo make run-notebook

lint:
	flake8 main.py

format:
	black main.py

run:
	uv run main.py

run-notebook:
	marimo run hello-marimo-smoldocling.py

help:
	@echo make run

lint:
	flake8 main.py

format:
	black main.py

run:
	uv run main.py

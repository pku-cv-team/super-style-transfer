source_dir = style_transfer
test_dir = tests

.PHONY: test lint typecheck format clean

test:
	pytest

lint:
	pylint $(source_dir) $(test_dir)

format:
	black $(source_dir) $(test_dir)

clean:
	rm -rf .tox .nox .pytest_cache .mypy_cache __pycache__

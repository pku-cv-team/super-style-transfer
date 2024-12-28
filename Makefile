source_dir = style_transfer
test_dir = tests

.PHONY: test lint typecheck format clean gatys

test:
	pytest

lint:
	pylint $(source_dir) $(test_dir)

format:
	black $(source_dir) $(test_dir)

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +

gatys:
	python style_transfer/train.py --config experiments/config_gatys.json

lapstyle:
	python style_transfer/train.py --config experiments/config_lapstyle.json

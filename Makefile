clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style

##### basic #####
set-dev:
	pip3 install -r requirements.txt

test:
	pip3 install pytest==6.2.4
	python3 -m pytest tests/

set-style-dep:
	pip3 install isort==5.9.3 black==21.7b0 flake8==3.9.2

set-style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .
	flake8 .

check-quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .
	flake8 .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache

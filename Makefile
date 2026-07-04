.DEFAULT_GOAL := check

.PHONY: install train evaluate test serve dashboard docker lint format format-check check clean

install:
	pip install -r requirements.txt

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

test:
	pytest -v

lint:
	ruff check src/ tests/

format:
	black src/ tests/

format-check:
	black --check src/ tests/

check: lint format-check test

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/dashboard/app.py

docker:
	docker build -t student-success-analytics .

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('.ruff_cache', ignore_errors=True)"
	python -c "import shutil; shutil.rmtree('htmlcov', ignore_errors=True)"
	python -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"
	python -c "import shutil; shutil.rmtree('build', ignore_errors=True)"

data:
	python -m src.pipeline

train: data
	python -m src.training
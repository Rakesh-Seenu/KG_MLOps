.PHONY: setup download preprocess train evaluate serve clean

## ── Setup ───────────────────────────────────────────────────────────────────
setup:
	@echo "✅ Setting up project..."
	pip install -r requirements.txt

## ── Data ────────────────────────────────────────────────────────────────────
download:
	@echo "📥 Downloading DRKG dataset..."
	python data/download_drkg.py

preprocess:
	@echo "⚡ Running RAPIDS GPU preprocessing..."
	python data/preprocess.py

## ── Training ────────────────────────────────────────────────────────────────
train:
	@echo "🚀 Starting training with PyTorch Lightning..."
	python training/trainer.py

evaluate:
	@echo "📊 Evaluating model..."
	python training/evaluate.py

## ── Serving ─────────────────────────────────────────────────────────────────
serve:
	@echo "🌐 Starting FastAPI server..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	@echo "🎨 Launching Gradio demo..."
	python app.py

## ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t biokg-predictor .

docker-run:
	docker run -p 8000:8000 --gpus all biokg-predictor

## ── Clean ────────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

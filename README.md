# LLM From Scratch

A minimal, educational implementation of a Large Language Model (LLM) inspired by LLaMA, written in pure Python and PyTorch. This project demonstrates the core components of transformer-based language models, including RMSNorm, Rotary Embeddings (RoPE), SwiGLU activation, and multi-head attention, trained on the Tiny Shakespeare dataset.

---

## Features

- **Character-level tokenizer** for simplicity and transparency
- **RMSNorm** for efficient normalization
- **Rotary Embeddings (RoPE)** for positional encoding
- **SwiGLU activation** for improved expressiveness
- **Multi-head masked attention**
- **Configurable model size and hyperparameters**
- **Dockerized for easy deployment**

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/llm-from-scratch.git
cd llm-from-scratch
```

### 2. Install dependencies

#### With pip

```bash
pip install -r requirements.txt
```

#### Or with Docker

```bash
docker build -t llm-from-scratch .
```

### 3. Run training and generation

#### Locally

```bash
python main.py
```

#### With Docker

```bash
docker run --rm llm-from-scratch
```

---

## Project Structure

```
llm-from-scratch/
├── config.py           # Model and training configuration
├── data.py             # Data loading, preprocessing, batching
├── model.py            # Model components and architecture
├── train.py            # Training, evaluation, and generation utilities
├── main.py             # CLI entrypoint
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Build and packaging info
├── Dockerfile          # Docker image definition
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Customization

- **Change model size or hyperparameters:** Edit `config.py`.
- **Use a different dataset:** Modify the URL in `main.py` and adjust preprocessing in `data.py` if needed.
- **Experiment with architecture:** Tweak or extend the model in `model.py`.

---

## Citation & Credits

- Inspired by [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Tiny Shakespeare dataset from [Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn)

---

## Deployment

- All core modules (`config.py`, `data.py`, `model.py`, `train.py`, `main.py`) are present.
- Dependencies are listed in `requirements.txt` and `pyproject.toml`.
- The Dockerfile is production-ready.
- Use the following commands to build and run:

```bash
docker build -t llm-from-scratch .
docker run --rm llm-from-scratch
```

Or run locally:

```bash
pip install -r requirements.txt
python main.py
```

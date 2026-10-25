# Local run (without Docker)

Python 3.10+. GPU optional — every number in the committed report was produced on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cpu wheel
pip install -r requirements.txt
export HF_HOME="$PWD/hf-cache"
```

```bash
python scripts/minkpp.py
python scripts/01_cache.py --self-check

# Skip if using committed caches in cache/
python scripts/01_cache.py --model state-spaces/mamba-1.4b-hf --all-splits --device cuda
# or CPU: omit --device / use --device cpu

python scripts/02_aggregate.py --model mamba-1.4b-hf --n-boot 2000
python scripts/03_figures.py --model mamba-1.4b-hf
python scripts/build_notebooks.py --execute
```

`mamba-ssm` / `causal-conv1d` are **not** required: `transformers` uses a reference PyTorch scan.

### Runtimes (12 CPU threads, float32) — from the original practical record

| Model | Split | Wall clock |
|---|---|---:|
| pythia-160m | 32/64/128 | ~6 min total |
| mamba-1.4b-hf | 32/64/128 | ~1h45 total |
| pythia-6.9b | needs ~14 GB — GPU + float16 | ~10 min on a 20 GB card |

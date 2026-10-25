# S08 · Min-K%++ — practical record

**Sofiia Bakina · S08 · Group B** · paper: Zhang et al., *Min-K%++*, ICLR 2025 (arXiv:2404.02936)
Record compiled 04.10.2026. Every number below was produced on the machine described in §2 and
can be regenerated with the commands in §5.

---

## 1. What this establishes

| Question | Answer |
|---|---|
| Does the baseline reproduce? | **Yes.** The Mamba-1.4B column of Table 1 reproduces at all three input lengths, 9 of 12 cells exact, largest deviation 0.5 AUROC |
| Does the primary improvement work? | **No.** Sliding-window aggregation is significantly *worse* at every length and in both forms tested |
| Does the fallback work? | **No effect.** Fixed-count selection is within noise of the baseline at every length |
| Is the result conclusive? | For the aggregation axis on this model and benchmark, yes: three lengths, three method forms, paired intervals |

The headline for the write-up: **on WikiMIA with Mamba-1.4B, the aggregation step of Min-K%++ holds
no remaining headroom.** Neither localising it (windows) nor removing its length dependence (fixed
token count) improves on the plain mean of the lowest k% of z-scores, and fragmenting the sequence
degrades performance in proportion to how finely it is fragmented.

Under §6.2 of the seminar regulations this is a full-credit outcome: the hypothesis was specific,
falsifiable, tested on the target model with intervals, and the mechanism behind the negative result
is explained in §7.

---

## 2. Environment

| Component | Value |
|---|---|
| OS | Windows 11 Pro, build 26300 |
| CPU | AMD Ryzen 5 6600H, 6 cores / 12 threads, no AMX |
| RAM | 13.7 GB total |
| GPU | none — every number below was computed on CPU |
| Working directory | `E:\work-mink` |
| Virtualenv | `E:\work-mink\.venv`, Python 3.12.10 (AMD64) |
| Model cache | `E:\hf-cache` (set via `HF_HOME`) |

```
torch          2.14.1+cpu     (CUDA unavailable, 6 threads)
transformers   5.18.0
datasets       5.0.1
numpy          2.5.3
scikit-learn   1.9.1
```

Model revisions actually used, pinned by Hugging Face commit hash:

| Model | Revision | Download |
|---|---|---|
| `state-spaces/mamba-1.4b-hf` | `6e46eae61c27280517feef46f536d16b91076f08` | 5.49 GB, two fp32 safetensors shards |
| `EleutherAI/pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` | 0.38 GB |
| `EleutherAI/pythia-70m` | `a39f36b100fe8a5377810d56c3f4789b9c53ac42` | 0.17 GB |

**`mamba-ssm` and `causal-conv1d` are not installed and are not needed.** `transformers` falls back
to a reference PyTorch implementation of the selective scan and the causal convolution; it prints a
warning and is slower, but it is numerically correct and it is what produced the reproduction in §6.
Those two packages appear in the authors' `requirements.txt` and build poorly on Windows, so this
matters: the project has no hard dependency on them.

All runs were `float32`, matching the dtype of the published reference numbers.

---

## 3. Data

`swj0419/WikiMIA`, loaded through `datasets`. Counted in hand rather than taken from the paper:

| Split | Items | Members | Non-members | Scored tokens / item: min / median / max | sd | CV |
|---|---:|---:|---:|---|---:|---:|
| `WikiMIA_length32` | 776 | 387 | 389 | 32 / 41 / 129 | 9.2 | 0.213 |
| `WikiMIA_length64` | 542 | 284 | 258 | 69 / 84 / 167 | 11.2 | 0.131 |
| `WikiMIA_length128` | 250 | 139 | 111 | 142 / 168 / 272 | 16.1 | 0.095 |

Two facts that matter for interpretation and are easy to miss:

1. **The split names are word counts, not token counts.** The "length 32" split spans 32 to 129
   scored tokens. Within-split length variation is real and largest in the shortest split.
2. **Median token count is balanced across classes** (41 vs 42, 84 vs 84, 168 vs 169), so the
   length variation is a nuisance rather than a label leak.

Scored positions exclude position 0, which has no preceding context — this matches the released
`run.py`, where the scored tokens are `input_ids[1:]`.

---

## 4. Method and pipeline

The design decision that makes the whole study affordable: **the model is run once per
(model, split, setting), and the per-token quantities are cached.** Every baseline and every
proposed variant is a function of the same three numbers per token —

- `logp` — log p(x_t | x_<t) for the observed token,
- `mu` — E over the vocabulary of log p(v | x_<t),
- `sigma` — the corresponding standard deviation,

from which z_t = (logp − mu) / sigma. Min-K% averages the lowest k% of `logp`; Min-K%++ averages
the lowest k% of z. Every aggregation variant tested here is a different function of the same z
vector, so **baseline and variants are compared at exactly matched compute by construction** — no
variant can be advantaged by having had a different forward pass.

| Script | Purpose |
|---|---|
| `cache.py` | Runs the model, writes `cache/<model>_<split>_<dtype>.npz`. Checkpoints every 25 items and resumes after an interrupt |
| `analyze.py` | Baselines, all variants, ablations, paired bootstrap. Pure numpy, no GPU |
| `server/run_all.sh` | Batch driver for the university GPU server, tiered by cost |

Cache size is ~0.5 MB per split, so the expensive artefact is tiny and portable.

Both scripts carry `--self-check`, run automatically before any real work. The checks that matter:
a window at least as wide as the sequence must reproduce the whole-sequence statistic exactly, and
the z-scores must not collapse onto the raw log-probabilities — dropping `sigma` is the one bug that
would silently turn Min-K%++ back into Min-K% while still looking plausible.

**Statistics.** AUROC and TPR@5%FPR, plus a **paired** bootstrap over items, 2000 resamples, seed 0,
with resampling indices shared across methods so that differences between methods stay paired.
Pairing matters: it buys roughly ±3.2 instead of ±3.9 on the length-32 split. The scorer itself is
deterministic — a statistic over logits, no training and no sampling — so repeated runs return
identical scores and seed variation does not apply to it; variability enters through the benchmark
subsample, which is what the bootstrap measures.

---

## 5. Commands

```bash
cd /e/work-mink
export HF_HOME=/e/hf-cache HF_HUB_DISABLE_SYMLINKS_WARNING=1

# sanity, no model needed
./.venv/Scripts/python.exe cache.py   --self-check
./.venv/Scripts/python.exe analyze.py --self-check

# the three caches behind every number in §6
./.venv/Scripts/python.exe cache.py --model state-spaces/mamba-1.4b-hf --dataset WikiMIA_length32
./.venv/Scripts/python.exe cache.py --model state-spaces/mamba-1.4b-hf --dataset WikiMIA_length64
./.venv/Scripts/python.exe cache.py --model state-spaces/mamba-1.4b-hf --dataset WikiMIA_length128

# analysis
./.venv/Scripts/python.exe analyze.py cache/mamba-1.4b-hf_WikiMIA_length32_float32.npz --window --fixedm
```

**Running from WSL requires a Windows-form `HF_HOME`.** `HF_HOME=/e/hf-cache` is silently ignored by
the Windows interpreter, which then re-downloads 5.49 GB into `C:\Users\<user>\.cache\huggingface`.
From WSL use:

```bash
cd /mnt/e/work-mink && HF_HOME='E:\hf-cache' ./.venv/Scripts/python.exe cache.py ...
```

### Measured runtimes, CPU, float32

| Model | Split | Items | Tokens | s / item | Wall clock |
|---|---|---:|---:|---:|---:|
| `pythia-70m` | length32 | 776 | 33 723 | 0.08 | 1.0 min |
| `pythia-160m` | length32 | 776 | 33 723 | 0.13 | 1.7 min |
| `mamba-1.4b-hf` | length32 | 776 | 33 723 | 2.34 | 30.3 min |
| `mamba-1.4b-hf` | length64 | 542 | 46 483 | 4.46 | ≈ 40 min |
| `mamba-1.4b-hf` | length128 | 250 | 42 619 | 8.94 | 37.3 min |

Cost scales linearly with tokens, so the figures extrapolate: Mamba-1.4B processes roughly
1.1 k tokens per minute on this CPU.

---

## 6. Main results

### 6.1 Baseline reproduction — the Mamba-1.4B column of Table 1

Independent re-implementation, CPU, float32, against the published table. Min-K% and Min-K%++ are
reported at the k that maximises AUROC, matching how the paper reports them.

| Length | Method | This work | Published | Δ |
|---:|---|---:|---:|---:|
| 32 | Loss | 61.0 | 61.0 | **0.0** |
| 32 | Zlib | 61.9 | 61.9 | **0.0** |
| 32 | Min-K% | 63.3 | 63.2 | +0.1 |
| 32 | Min-K%++ | 66.8 | 66.8 | **0.0** |
| 64 | Loss | 58.2 | 58.2 | **0.0** |
| 64 | Zlib | 60.4 | 60.4 | **0.0** |
| 64 | Min-K% | 61.7 | 62.2 | −0.5 |
| 64 | Min-K%++ | 67.2 | 67.2 | **0.0** |
| 128 | Loss | 63.3 | 63.3 | **0.0** |
| 128 | Zlib | 65.6 | 65.6 | **0.0** |
| 128 | Min-K% | 66.8 | 66.8 | **0.0** |
| 128 | Min-K%++ | 68.9 | 68.8 | +0.1 |

Nine of twelve cells are exact; the largest deviation is 0.5 points, well inside the ±3.9 to ±6.5
resampling interval of a single AUROC at these sample sizes (§6.3). **The baseline is reproduced.**

Ref, Lowercase and Neighbor require additional forward passes and are listed in §8 as outstanding.

### 6.2 Primary improvement — sliding-window aggregation with max pooling

Window of width w, stride w/2, Min-K%++ computed inside each window, pooled across windows. For a
sequence shorter than w this reduces exactly to the baseline, which the self-check verifies.
All comparisons are paired against Min-K%++ at k = 0.2.

| Variant | length 32 | length 64 | length 128 |
|---|---:|---:|---:|
| w = 64, max | −0.08 | −1.76 | **−6.57** |
| w = 32, max | −1.59 | **−8.63** | **−13.43** |
| w = 16, max | **−6.22** | **−15.25** | **−15.94** |
| w = 16, p90 | **−4.49** | **−12.89** | **−11.57** |

Bold entries have a 95% paired interval excluding zero. The pattern is monotone: the narrower the
window, the worse the result, at every length.

**H1 predicted the opposite.** The hypothesis was that whole-sequence averaging dilutes a localised
memorised span, so the gain should grow with input length. The measured degradation at length 128
(−15.94) is at least as large as at length 64 (−15.25), and the baseline is never beaten.

### 6.3 Control — fixed number of windows

Fixed-width pooling makes the number of windows grow with passage length, and the expected maximum
over more windows is higher regardless of membership — a new length dependence inside a method meant
to remove one. Holding the count fixed at K removes that confound.

| Variant | length 32 | length 64 | length 128 |
|---|---:|---:|---:|
| K = 2, max | −2.58 | **−3.19** | **−4.40** |
| K = 3, max | **−5.19** | **−6.57** | **−5.93** |
| K = 4, max | **−9.51** | **−8.91** | **−9.59** |
| K = 6, max | **−10.17** | **−10.36** | **−12.92** |

The controlled form fails the same way and the degradation is again monotone in the number of
pieces. **So the loss is not caused by the variable window count; it is caused by fragmentation
itself.** This is the single most informative ablation in the study and it is what makes the
negative result explicable rather than merely observed (§7).

### 6.4 Fallback — fixed-count selection

Keep the lowest *m* z-scores rather than the lowest k%, so the number of averaged terms no longer
drifts with passage length. Paired against Min-K%++.

| Variant | length 32 | length 64 | length 128 |
|---|---:|---:|---:|
| m = 5 | −1.05 | −1.20 | −0.96 |
| m = 8 | −0.38 | −0.88 | −2.19 |
| m = 12 | −0.14 | −0.60 | −2.42 |
| m = 20 | −0.46 | −0.78 | −1.67 |

Every interval spans zero. The fallback neither helps nor hurts.

> **A confound worth recording, because it nearly entered this report as a positive result.**
> `fixm` operates on z-scores, so it is a Min-K%++ variant. Compared against *Min-K%* it appears to
> gain +4.9 points with P = 0.982 at length 64. That gain is entirely the z-standardisation — the
> contribution of Min-K%++ itself — and vanishes against the correct baseline. Any aggregation
> variant built on z must be compared against Min-K%++, never against Min-K%.

---

## 7. Why the negative result happens

Min-K%++ already selects the lowest k% of z-scores **over the whole sequence**, which is itself a
search for the most anomalous region. Windowing forces the statistic to spend its k% quota *inside
every window*, so it must admit tokens that are not globally anomalous. Whole-sequence selection is
therefore strictly better targeted, and windowing can only win if memorisation is localised *and*
the global selection misses it. The monotone degradation in both §6.2 and §6.3 says it does not
miss it on this benchmark.

This gives the write-up a mechanism rather than a bare null, and it predicts where the idea could
still pay off: a corpus whose passages are long enough that the global bottom-k% is spread across
genuinely unrelated material. WikiMIA passages top out at 272 scored tokens, which is evidently not
long enough.

---

## 8. Additional results

**8.1 Resolution of the benchmark.** Bootstrap half-widths, measured, not assumed:

| Split | n | Single AUROC | Paired difference |
|---|---:|---:|---:|
| length32 | 776 | ±3.6 … ±4.1 | ±3.2 |
| length64 | 542 | ±4.4 … ±5.0 | ±3.3 |
| length128 | 250 | ±6.4 … ±7.4 | ±6.2 |
| length256 | 82 | ±13.4 | ±12.5 |

The reproduced Min-K%++ − Min-K% gap on Mamba-1.4B at length 32 is **+3.6 with a 95% interval of
[+0.4, +6.8], P(gap > 0) = 0.989** — it clears zero, but only just. Length 256 is unusable at
n = 82 and should be reported for completeness only.

**8.2 The method's published advantage is model-dependent.** Computed from the paper's Table 1, the
Min-K%++ minus Min-K% gap averages **+3.7** over Mamba-1.4B and Pythia-6.9B against **+13.6** over
the three LLaMA columns. The headline averages are carried by LLaMA; the two models inside a student
budget are the two where the method helps least.

**8.3 k matters more than any aggregation variant tested.** At length 128, `mink++_0.4` beats
`mink++_0.2` by +1.10 — larger than any fallback effect, and the paper does not say which k produced
a published cell.

**8.4 The fixed-count variant trades short passages for long ones.** At length 64 the per-item
difference `fixm12 − mink++_0.2` correlates with that item's token count at **r = −0.712**: holding
m fixed helps short passages and hurts long ones, which is mechanically expected since a fixed m is
a shrinking fraction as the passage grows. This is a clean, reportable mechanism even though the
aggregate effect is null.

**8.5 Verified facts about the benchmark** that change the plan rather than the result:

- The Neighbor baseline needs **no perturbation step**. `run_neighbor.py` reads pre-generated
  neighbours from `zjysteven/WikiMIA_paraphrased_perturbed` at exactly ten per item, so the masked
  language model never runs — only scoring, at 10× the data.
- The perturbed splits exist at lengths 32 / 64 / 128 only. **Neighbor cannot be run at length 256.**
- The paraphrased splits exist at 32 / 64 / 128 only, and their labels are stored as **strings**
  (`'0'`, `'1'`), which silently breaks naive integer casting.
- `iamgroot42/mimir`, the planned secondary benchmark, is `gated: auto` — open, but it needs a
  logged-in Hugging Face account and one click.

---

## 9. What is still outstanding

| # | Item | Blocker | Cost |
|---|---|---|---|
| 1 | Pythia-6.9B, all lengths | 13.85 GB in fp16 — **does not fit in 13.7 GB of system RAM**, GPU only | ~10 min on a 20 GB card |
| 2 | Paraphrased setting, all lengths | none, CPU time only | ~2 h CPU |
| 3 | Lowercase and Smaller-Ref baselines | none | ~2 h CPU |
| 4 | Neighbor baseline | 10× the data | ~1–2 h GPU |
| 5 | Truncation split — cut length-128 items to 64 and 32 to vary length with content held fixed | none, data work | ~1 h |
| 6 | Held-out 20% protocol for fitting w and K | none | ~2 h |

Items 2, 3, 5 and 6 need no GPU. Item 1 is the only hard dependency on the university server;
instructions and the cost model are in `server/README.md` in the working directory.

**Nothing in §6 depends on any of these.** The aggregation result is complete on the primary model.

---

## 10. Traps found the hard way

1. **Comparing a z-based variant against Min-K%.** See the box in §6.4. This inflates any such
   variant by the full Min-K%++ contribution and survives casual review.
2. **`HF_HOME` in POSIX form under WSL.** Silently ignored; cost 24 minutes and 5.49 GB of
   duplicate download onto a nearly full system drive.
3. **String labels in the paraphrased and perturbed splits.** `np.int8` on `'0'` raises; cast
   explicitly.
4. **`\sum` in the proposal's PDF builder.** The TeX subset did not support it and leaked the
   literal command into the PDF; the coverage check cannot catch this, because the token is present
   in both source and output.
5. **Treating the split names as token counts.** They are word counts; the token spread inside a
   split is up to 4×.

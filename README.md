# int2int2

Accompanying code for Appendix B (*Transformers*) of

> B. S. Banwait, X. Huang, K.-H. Lee, S. Lee, T. Oliver, A. Pozdnyakov,
> *Decision trees, Frobenius traces, and Weierstrass coefficients of elliptic curves*,
> [arXiv:2607.24251](https://arxiv.org/abs/2607.24251)

A deliberately minimal encoder–decoder transformer, in the style of
[Int2Int](https://arxiv.org/abs/2502.17513), that reads the Frobenius traces
of an elliptic curve and predicts the reduced minimal Weierstrass coefficients
`(w1, w2, w3)`.

## Results

Validation accuracy on `E_2` — all 2,100,011 isogeny classes of conductor
`< 5 x 10^5` from the LMFDB — under greedy decoding. Reproduces Table 11.

```
$ python evaluate.py --session cache/w1w2w3

                      w1        w2        w3     joint
transformer       100.0%     89.2%     80.5%     72.6%
baseline           50.8%     36.2%     65.3%
```

`baseline` is the majority-class rate for that coefficient. The first
experiment, `(a_p)_{p<100} -> (w1, w2)`, reaches 100%:

```
$ python evaluate.py --session cache/w1w2

                      w1        w2     joint
transformer       100.0%    100.0%    100.0%
baseline           50.8%     36.2%
```

Those are measured on 10,000 validation classes, the `eval_size` cap the
published runs used. `--eval-size 0` scores the entire 20% split (420,003
classes) instead, which moves `w1w2w3` by up to a point:

```
$ python evaluate.py --session cache/w1w2w3 --eval-size 0 --batch-size 512

                      w1        w2        w3     joint
transformer       100.0%     88.8%     81.6%     73.3%
baseline           51.2%     35.8%     65.8%
```

`w1w2` is 100% on the full split too. See the note on `eval_size` below.

## Quickstart

```bash
pip install -r requirements.txt

python data/build_dataset.py                       # ~190 MB download, a few minutes
python train.py    --config configs/w1w2w3.yaml
python evaluate.py --session cache/<your session>
```

`train.py` writes each run to `cache/sesh_<timestamp>/` (model, optimiser
state, the exact config used, and `training.log`). Pass `--resume
cache/sesh_<timestamp>` to continue one.

## The data

`data/build_dataset.py` rebuilds both parquets from
[ecdata](https://github.com/JohnCremona/ecdata), Cremona's database of all
elliptic curves over **Q** of conductor below 500000, which is the source
LMFDB itself is built from. It downloads only the two directories it needs
(`allcurves`, `alllabels`, ~190 MB together), not the full 1.8 GB of
`curvedata`.

| file | input | target |
|---|---|---|
| `aps100_ainvs.parquet` | `a_2, a_3, ..., a_97` (25 traces) | `w1, w2` |
| `aps100_ainvs_full.parquet` | `a_2, ..., a_97, N` | `w1, w2, w3` |

One row per isogeny class. The representative is curve number **1 in the LMFDB
numbering**, which differs from Cremona's (Cremona `100a3` is LMFDB `100.a1`) —
hence the `alllabels` join. `(w1, w2, w3)` are the first three a-invariants of
that curve's global minimal model.

LMFDB does not store `a_p`, so `src/data/frobenius.py` recomputes it by point
counting:

> `a_p = p - A`, where `A` is the number of affine points of the reduced
> minimal model over `F_p`.

One formula covers both cases — for good `p`, `#E(F_p) = A + 1`; for bad `p`,
the singular point drops out and infinity comes in, so `#E_ns(F_p) = A` —
and it yields `a_p = 0, +1, -1` for additive, split and non-split reduction
respectively.

## Model and tokenisation

Following Appendix B.1: one encoder layer, one decoder layer, one attention
head, `d_model = 128`, `d_ff = 256`, no dropout, learnable positional
embeddings (~0.36 M parameters). Adam, learning rate `1e-4`, batch size 32,
80/20 train/validation split. Weight decay is `1e-2` for `w1w2w3` and `0`
for `w1w2` — see the note below.

Integers are written in base 30 as a sign token (`+`/`-`) followed by base-30
digits; with `<bos> <eos> <pad> <sep> <unk>` that is a vocabulary of 37. A
sequence of integers is the concatenation of those encodings between a leading
`<bos>` and a trailing `<eos>`.

## Layout

```
train.py                      train from a config
evaluate.py                   per-coefficient + joint accuracy (Table 11)
configs/w1w2.yaml             (a_p)_{p<100}      -> (w1, w2)
configs/w1w2w3.yaml           ((a_p)_{p<100}, N) -> (w1, w2, w3)
data/build_dataset.py         rebuild the parquets from ecdata
src/data/frobenius.py         a_p by point counting
src/data/tokenizer.py         base-b sign/digit tokenizer
src/data/dataset.py           tokenising + the 80/20 split
src/model/transformer.py      the model
src/training/trainer.py       training loop
src/training/evaluator.py     validation metrics
src/utils/transformer_analysis.py   attention / embedding plots
Analysis.ipynb                worked examples of the above
cache/w1w2, cache/w1w2w3      the two checkpoints behind the results
```

## Notes

- **Decoding is greedy.** `Transformer.generate` takes the argmax by default.
  Sampling from the softmax instead (`do_sample=True`) measures a random draw
  from the model and scores several points below the reported numbers — on
  `cache/w1w2w3` it gives a joint accuracy of 0.66 rather than 0.726.
- **Padding is not masked.** Attention sees `<PAD>` positions and the
  cross-entropy is taken over them too. This is how the reported models were
  trained; adding a mask changes the numbers and invalidates the checkpoints.
- **The two experiments used different weight decay.** Appendix B.1 states
  `1e-2`, which is what `configs/w1w2w3.yaml` uses and what reproduces
  Table 11. The 100% result on `(w1, w2)`, however, came from a run with
  weight decay `0`, and `configs/w1w2.yaml` is set accordingly. At `1e-2` that
  model converges by epoch 2 to a training loss of ~0.053 and then oscillates
  between 88% and 91% for at least 29 epochs; at `0` it reaches a training
  loss of 0.011 and exactly 100% validation accuracy within a single epoch.
  Weight decay appears to stop this model from representing the exact rule
  `(w1, w2) = f(a_2, a_3)`.
- **Table 11 is measured on a 10,000-class subsample of the validation set.**
  Appendix B.1 describes an 80/20 split and reports accuracy over the
  validation classes; both configs additionally set `eval_size: 10000`,
  because autoregressive decoding over all 420,003 is slow. The cap is a
  random subsample, so the estimates are unbiased, but at n=10,000 one
  standard error is ~0.4 points: on the full split `w3` is 81.6% rather than
  80.5% (~2.8 sigma) and the joint figure is 73.3% rather than 72.6%. Every
  qualitative claim in B.2 survives — `w1` exact, `w2` high but imperfect,
  `w3` the bottleneck, and `w3` still above the 76.68% decision-tree result.
- `num_epochs` is 150 in both configs, but neither run needed it: `w1w2`
  saturates at 100% in one epoch, and `w1w2w3` was stopped after 16 epochs.
  `evaluate.py` reads `best_model.pth`, the lowest-validation-loss epoch.
- `encoder_only` and `decoder_only` exist in `Transformer` but are not used by
  either experiment and are not covered by the results above.

## Citation

```bibtex
@misc{int2int2,
  author = {Huang, Xiaoyu},
  title  = {Int2int2},
  year   = {2026},
  howpublished = {\url{https://github.com/cocoxhuang/int2int2}}
}
```

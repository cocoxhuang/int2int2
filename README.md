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
state, the exact config used, and `training.log`). Pass `--resume cache/sesh_<timestamp>` to continue one.

## The data

`data/build_dataset.py` rebuilds both parquets from
[ecdata](https://github.com/JohnCremona/ecdata), Cremona's database of all
elliptic curves over **Q** of conductor below 500000, which is the source
LMFDB itself is built from. It downloads only the two directories it needs
(`allcurves`, `alllabels`, ~190 MB together).


| file                        | input                             | target       |
| --------------------------- | --------------------------------- | ------------ |
| `aps100_ainvs.parquet`      | `a_2, a_3, ..., a_97` (25 traces) | `w1, w2`     |
| `aps100_ainvs_full.parquet` | `a_2, ..., a_97, N`               | `w1, w2, w3` |


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


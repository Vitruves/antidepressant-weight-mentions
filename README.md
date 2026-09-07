# antidepressant-weight-mentions

Annotated corpus and encoder benchmark for detecting **patient-reported mentions of weight change** in online antidepressant reviews.

A review is assigned one of three labels (**weight gain**, **weight loss**, or **no reference to weight change**), and the label is applied only when the writer attributes the change to the drug the review is about.

> **What this is not.** A positive prediction means the text contains a mention. It is not evidence that an adverse drug reaction occurred, that body weight actually changed, or that the drug caused the change. Treat the output as a screening signal over large narrative corpora, not as pharmacovigilance evidence.

## Data

The two annotated corpora are **not stored in this repository**. They are deposited on FigShare, which carries the DOI, the versioning and the data licence (see `data/README.md`). Download both files into `data/` before running the commands below.

Reviews were collected in 2022 from Drugs.com, WebMD, Everyday Health and Ask a Patient. The two sets are disjoint; they share no review.

| File | n | Molecules | No mention | Gain | Loss |
|:---|---:|---:|---:|---:|---:|
| `..._training-corpus_n8000.csv` | 8,000 | 31 | 7,174 (89.7 %) | 621 (7.8 %) | 205 (2.6 %) |
| `..._evaluation-batch_n2010.csv` | 2,010 | 19 | 1,762 (87.7 %) | 175 (8.7 %) | 73 (3.6 %) |

Columns: `ID`, `REVIEW`, `DRUG`, `WEIGHT_CHANGE` (0 = no mention, 1 = gain, 2 = loss). The `ID` prefix encodes the source site: `DC` Drugs.com, `WE` WebMD, `ED` Everyday Health, `AA` Ask a Patient.

The evaluation batch was annotated **after the training corpus was frozen**, on different reviews, and differs from it in molecule coverage, source mix and class prevalence. That is what makes the benchmark a measurement of transfer rather than of fit, so **do not merge the two sets** if you want comparable numbers.

Annotation was done by ten operators (two pharmacists, one general practitioner, two pharmacy students, five non-experts) under drug-expert supervision. 3,500 reviews were doubly annotated: 96 discrepancies (2.74 %), adjudicated by a single expert reviewer, Krippendorff's α = 0.841.

The Ask a Patient reviews were supplied by Askapatient.com to cover rare molecules, monoamine oxidase inhibitors and ATC classes the larger platforms barely reach. They are distinct from the public PsyTAR corpus, which was not used.

## Benchmark

21 pre-trained encoders (4.4 M to 434 M parameters, five architecture generations, general and biomedical pre-training) fine-tuned under one shared protocol and evaluated once on the 2,010-review batch.

AdamW, learning rate 2e-5, weight decay 0.01, linear schedule without warm-up, batch size 16, max length 512, dropout 0.1, class weights inversely proportional to frequency, at most 10 epochs, checkpoint selected on validation macro-F1, stratified 80/20 train/validation split with a fixed seed.

**Macro-F1 is the primary metric.** A majority-class classifier reaches 0.877 accuracy on this batch but only 0.311 macro-F1, so accuracy alone is close to uninformative.

| Model | Accuracy | Macro-F1 | κ | Recall (loss) |
|:---|---:|---:|---:|---:|
| answerdotai/ModernBERT-Large-Instruct | 0.990 | **0.964** | 0.955 | 0.932 |
| answerdotai/ModernBERT-large | 0.990 | 0.963 | 0.953 | 0.945 |
| answerdotai/ModernBERT-base | 0.982 | 0.938 | 0.919 | 0.959 |
| Alibaba-NLP/gte-large-en-v1.5 | 0.984 | 0.938 | 0.930 | 0.890 |
| WhereIsAI/UAE-Large-V1 | 0.977 | 0.926 | 0.900 | 0.918 |
| … | | | | |
| google/electra-base-discriminator | 0.932 | 0.579 | 0.707 | **0.000** |
| prajjwal1/bert-tiny | 0.925 | 0.558 | 0.693 | **0.000** |

Full table with confidence intervals and architecture metadata: `results/benchmark.csv`.

Three findings worth carrying over to other tasks:

- **Architecture generation beats scale.** ModernBERT-base (150 M) outperforms every first-generation encoder tested, including models twice its size.
- **Contrastive embedding backbones are efficient.** all-MiniLM-L6-v2, at 23 M parameters, is within 0.064 macro-F1 of a 434 M model.
- **Biomedical pre-training does not help here.** Patient reviews are written in lay register ("I packed on weight"), not the formal idiom domain models are trained on.

**Never report accuracy alone on this task.** The four weakest encoders sit within 5.5 accuracy points of the best models yet never predict the weight-loss class at all: recall exactly 0.000. Accuracy hides this completely; per-class recall does not.

## Layout

```
data/      empty; download the corpora from FigShare into it
scripts/   BERT_train.py         fine-tuning and evaluation
           fast_predictions.py   batched inference over a large table
results/   benchmark.csv         the 21 models, all metrics
           models_meta.csv       parameters, depth, context, family
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 to 3.12. See the header of `requirements.txt` for CUDA variants.

## Fine-tune and evaluate

```bash
python scripts/BERT_train.py \
  -i data/antidepressant-weight-mentions_training-corpus_n8000.csv \
  --test-file data/antidepressant-weight-mentions_evaluation-batch_n2010.csv \
  --text-col REVIEW --label-col WEIGHT_CHANGE \
  --hf-model-name answerdotai/ModernBERT-base \
  --epochs 10 --learning-rate 2e-5 --batch-size 16 \
  --use-class-weights --evaluation-metric f1_macro \
  -o runs/modernbert-base
```

Swap `--hf-model-name` for any of the 21 identifiers in `results/benchmark.csv` to reproduce a row of the table. `Alibaba-NLP/gte-large-en-v1.5` ships its architecture as remote code and additionally needs `--trust-remote-code`; the flag is off by default because it executes code downloaded from the Hub.

## Predict over a large file

```bash
python scripts/fast_predictions.py \
  -i reviews.parquet -o predictions.parquet \
  -m runs/modernbert-base --text-col REVIEW \
  --batch-size 128 --fp16 \
  --labels none gain loss --include-probabilities
```

Reads CSV, TSV, JSON, JSONL, Excel or Parquet. Use `--input-chunksize` (or `--auto-chunk`) for files that do not fit in memory; chunks are written to a temporary directory and merged at the end.

## Known limitations

- The evaluation batch comes from the same four platforms, the same collection window and the same annotation team as the training corpus. It is not an external cohort; platform-held-out and temporally held-out evaluation remain open.
- **The target drug name is not part of the model input.** The dominant residual error is a review reporting weight change for a *different* drug than the one under review, and the model has no direct way to resolve that attribution. Appending the molecule to the input is the obvious next experiment.
- Reviewers are self-selected and their reviews carry no verified demographic or clinical data. Any downstream epidemiological use inherits that selection.

## Citation

<!-- Replace with the published reference once available. -->
```bibtex
@article{yokoyama_weight_mentions,
  author  = {Yokoyama, Ta{\"i}oh and Natter, Johan and Godet, Julien},
  title   = {Detecting patient-reported mentions of weight change in online
             antidepressant reviews: an annotated corpus and a benchmark of
             21 transformer encoders},
  year    = {2026}
}
```

Dataset DOI: [10.6084/m9.figshare.33456934](https://doi.org/10.6084/m9.figshare.33456934) (FigShare). Embargoed until publication.

## Licence

- **Code** (`scripts/`, `results/`): Apache License 2.0, see `LICENSE`.
- **Data**: Creative Commons Attribution 4.0 International (CC BY 4.0), distributed with the FigShare deposit rather than with this repository.

The data licence covers the annotations and the compilation of the corpora. The review texts were written by the patients who posted them and were collected from publicly accessible pages; that authorship is unaffected by the licence under which this work is released.

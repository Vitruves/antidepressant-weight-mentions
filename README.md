# Detecting patient-reported mentions of weight change in online antidepressant reviews: an annotated corpus and a benchmark of 21 transformer encoders

[![Data DOI](https://img.shields.io/badge/data-10.6084%2Fm9.figshare.33456934-blue)](https://doi.org/10.6084/m9.figshare.33456934) [![Code licence](https://img.shields.io/badge/code-Apache%202.0-green)](LICENSE) [![Data licence](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

Code and results accompanying the article:

> Yokoyama T, Natter J, Godet J. **Detecting patient-reported mentions of weight change in online antidepressant reviews: an annotated corpus and a benchmark of 21 transformer encoders.** Manuscript under review, 2026.

**Authors.** Taïoh Yokoyama¹, Johan Natter¹, Julien Godet¹,²,³

¹ Faculté de Pharmacie, Université de Strasbourg, 67400 Illkirch, France<br>
² ICube — IMAGeS, UMR 7357, CNRS, Université de Strasbourg, 67400 Illkirch, France<br>
³ Groupe Méthodes en Recherche Clinique, Hôpitaux Universitaires de Strasbourg, 67000 Strasbourg, France

**Annotators.** Maxime Alter, Furkan Erol, Ahmed Guendouz, Vitoria Morais-Brazil, Paule Nkeng, Inas Oulkaid-Mouaddan, Sofia Salaa, Chinar Salmanli

**Correspondence.** julien.godet@unistra.fr

## Contents

- [Abstract](#abstract)
- [Key points](#key-points)
- [Task definition](#task-definition)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Reproducing the benchmark](#reproducing-the-benchmark)
- [Limitations](#limitations)
- [Citation](#citation)
- [Contributors and acknowledgements](#contributors-and-acknowledgements)
- [Funding and competing interests](#funding-and-competing-interests)
- [Licence](#licence)

## Abstract

**Objective.** Online drug reviews are a large, under-used source of patient-reported experience. We asked how reliably compact transformer encoders can identify, in free-text antidepressant reviews, whether the writer mentions weight gain, weight loss, or makes no reference to a change in body weight. The claim is deliberately restricted to mention detection: no attempt is made to confirm an adverse drug reaction, to quantify body weight, or to establish causality.

**Materials and methods.** A sample of 8,000 reviews, stratified by source website and by molecule, was drawn from 80,594 reviews of FDA-approved antidepressants collected from four consumer health websites. It covers 31 molecules and was annotated into three classes by ten operators under the supervision of drug experts (Krippendorff's α = 0.841, 95 % CI [0.803; 0.875], on 3,500 doubly annotated reviews). A label of weight gain or weight loss was applied only when the change was attributed by the writer to the drug the review was about. Models were trained on that corpus and evaluated on a separate batch of 2,010 reviews, annotated afterwards on different reviews under the same guideline, sharing no item with the training corpus and differing from it in molecule coverage (19 of 31 molecules), source mix and class prevalence. Twenty-one pre-trained encoders spanning 4.4 M to 434 M parameters, five architecture generations and two pre-training domains were fine-tuned under a single common protocol. Macro-F1 was defined a priori as the primary metric; accuracy with 95 % Wilson intervals, macro precision and recall, Cohen's κ and quadratic weighted κ are reported alongside.

**Results.** The evaluation batch contained 87.7 % reviews with no reference to weight change, 8.7 % weight gain and 3.6 % weight loss; a majority-class classifier reaches 0.877 accuracy but only 0.311 macro-F1. Across the 21 encoders, accuracy spanned a narrow band (0.924–0.990) whereas macro-F1 spanned 0.557–0.964 and cleanly separated four models that collapsed onto the majority class from seventeen that did not. The best model was ModernBERT-Large-Instruct (macro-F1 0.964, κ 0.955, accuracy 0.990, 95 % CI 0.985–0.994); ModernBERT-base (150 M parameters) outperformed every first-generation encoder tested, including models more than twice its size. Macro-F1 increased with model size (Spearman ρ = 0.79), but architecture generation mattered more than scale, and biomedical or clinical pre-training conferred no advantage. Per-class recall exposed the failure mode of the weakest encoders: all four never predicted weight loss at all (recall 0.000) while reaching 0.94–0.99 recall on weight gain, a pattern invisible in accuracy.

**Conclusion.** Modern transformer encoders identify patient-reported weight-change mentions in lay online narratives with high reliability and at negligible computational cost, which makes them suitable as a pre-screening layer over large review corpora. They do not, on their own, produce pharmacovigilance evidence: mention is not event, and drug attribution remains the dominant residual error. The contributions of this work are the annotated corpus and the reproducible benchmark rather than a new method.

**Keywords.** transformer encoders; text classification; antidepressants; body weight; patient-generated health data; pharmacovigilance signal screening

## Key points

- An annotated corpus of 8,000 online antidepressant reviews labelled for weight gain, weight loss or no reference to weight change, with a reproducible benchmark of 21 BERT-family encoders fine-tuned under one shared protocol.
- Performance is measured on a **separate batch of 2,010 reviews annotated after the training corpus was frozen**, with a different molecule mix and a higher prevalence of both minority classes. Each model saw it exactly once, so the benchmark measures transfer rather than fit.
- **Architecture generation matters more than scale.** ModernBERT-base (150 M) outperforms every first-generation encoder tested, including models twice its size, while biomedical and clinical pre-training bring no advantage on lay patient narratives.
- **Accuracy must not be reported alone on this task.** The four weakest encoders never predict the weight-loss class at all, yet sit within 5.5 accuracy points of the best models.
- **The dominant residual error is attribution**: weight change reported for a *different* drug than the one under review. The target molecule is not part of the model input, so this sub-problem remains open.

## Task definition

Each review is assigned exactly one of three labels: **weight gain**, **weight loss**, or **no reference to weight change**.

- The unit of judgement is the *text*, not the patient. Nothing in the annotation certifies that weight was measured, that the change was clinically significant, or that it would meet a case definition for an adverse drug reaction.
- The label is *drug-conditioned*. It is applied only when the writer attributes the change to the drug the review is about. A review describing weight change only for another drug ("*I gained 30 lbs on mirtazapine, so my doctor switched me to this one*") is labelled as no reference to weight change.

> **What a prediction means.** A positive prediction means the text contains a mention. It is not evidence that an adverse drug reaction occurred, that body weight actually changed, or that the drug caused the change. Treat the output as a screening signal over large narrative corpora, not as pharmacovigilance evidence.

## Data

The two annotated corpora are **not stored in this repository**. They are deposited on FigShare, which carries the DOI, the versioning and the data licence: [doi:10.6084/m9.figshare.33456934](https://doi.org/10.6084/m9.figshare.33456934) (embargoed until publication). Download both files into `data/` before running the commands below.

| File | n | Molecules | No mention | Gain | Loss |
|:---|---:|---:|---:|---:|---:|
| `antidepressant-weight-mentions_training-corpus_n8000.csv` | 8,000 | 31 | 7,174 (89.7 %) | 621 (7.8 %) | 205 (2.6 %) |
| `antidepressant-weight-mentions_evaluation-batch_n2010.csv` | 2,010 | 19 | 1,762 (87.7 %) | 175 (8.7 %) | 73 (3.6 %) |

Columns: `ID`, `REVIEW`, `DRUG`, `WEIGHT_CHANGE` (0 = no mention, 1 = gain, 2 = loss). The `ID` prefix encodes the source site: `DC` Drugs.com, `WE` WebMD, `ED` Everyday Health, `AA` Ask a Patient.

**Collection.** Reviews were collected between August and November 2022 from Drugs.com, WebMD, Everyday Health and Ask a Patient, giving 80,594 reviews of FDA-approved antidepressants from which the annotated samples were drawn. The Ask a Patient reviews were supplied by Askapatient.com to cover rare molecules, monoamine oxidase inhibitors and ATC classes the larger platforms barely reach. They are distinct from the public PsyTAR corpus, which was not used.

**Annotation.** Ten operators (two pharmacists, one general practitioner, two pharmacy students and five operators without drug-specific expertise) worked from a written guideline under drug-expert supervision. 3,500 of the 8,000 reviews were annotated twice, producing 96 discrepancies (2.74 %), all adjudicated by a single expert reviewer. Krippendorff's α over the doubly annotated subset was 0.841 (95 % CI [0.803; 0.875]).

**Two disjoint sets.** The evaluation batch was annotated after the training corpus was frozen, on different reviews, and differs from it in molecule coverage, source mix and class prevalence. **Do not merge the two sets** if you want numbers comparable to those reported here.

## Methods

**Models.** 21 publicly available pre-trained encoders, chosen to span scale (4.4 M to 434 M parameters), architecture generation (first-generation, compact or distilled, optimised successors, contrastive embedding backbones, ModernBERT family) and pre-training domain (general versus biomedical or clinical). Hub identifiers, parameter counts, depth and context length are in `results/models_meta.csv`.

**Fine-tuning protocol.** One protocol for every model: AdamW, learning rate 2e-5, weight decay 0.01, linear schedule without warm-up, batch size 16, maximum length 512 tokens, classification-head dropout 0.1, cross-entropy weighted by inverse class frequency, at most 10 epochs with early stopping, checkpoint selected on validation macro-F1, stratified 80/20 train/validation split with a fixed seed. Runs used a single NVIDIA RTX 3080 (10 GB), PyTorch 2.4 and Transformers 4.52.

**Evaluation.** Each model was evaluated once on the 2,010-review batch. **Macro-F1 is the primary metric**, defined a priori. A majority-class classifier reaches 0.877 accuracy on this batch but only 0.311 macro-F1 and κ = 0, so any model must be judged against that baseline, not against zero.

The uniform protocol is a deliberate choice with a known cost: a model that performs poorly under it may be under-tuned rather than unsuited to the task. The table below compares backbones *under one shared, moderate budget*; it does not estimate each model's ceiling.

## Results

Performance on the evaluation batch (n = 2,010; 1,762 / 175 / 73 reviews per class), ranked by macro-F1.

| # | Model | Params (M) | Accuracy | Macro-F1 | κ | Recall (gain) | Recall (loss) |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | answerdotai/ModernBERT-Large-Instruct | 395.9 | 0.990 | **0.964** | 0.955 | 0.954 | 0.932 |
| 2 | answerdotai/ModernBERT-large | 395.9 | 0.990 | 0.963 | 0.953 | 0.966 | 0.945 |
| 3 | answerdotai/ModernBERT-base | 149.7 | 0.982 | 0.938 | 0.919 | 0.926 | 0.959 |
| 4 | Alibaba-NLP/gte-large-en-v1.5 | 434.1 | 0.984 | 0.938 | 0.930 | 0.971 | 0.890 |
| 5 | WhereIsAI/UAE-Large-V1 | 335.1 | 0.977 | 0.926 | 0.900 | 0.960 | 0.918 |
| 6 | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract | 109.5 | 0.972 | 0.907 | 0.881 | 0.920 | 0.918 |
| 7 | FacebookAI/roberta-large | 355.4 | 0.970 | 0.903 | 0.873 | 0.966 | 0.877 |
| 8 | intfloat/e5-large-v2 | 335.1 | 0.971 | 0.899 | 0.876 | 0.960 | 0.863 |
| 9 | intfloat/e5-base-v2 | 109.5 | 0.968 | 0.889 | 0.862 | 0.926 | 0.849 |
| 10 | BAAI/bge-base-en-v1.5 | 109.5 | 0.967 | 0.888 | 0.858 | 0.909 | 0.849 |
| 11 | sentence-transformers/all-MiniLM-L6-v2 | 22.7 | 0.958 | 0.874 | 0.830 | 0.966 | 0.849 |
| 12 | medicalai/ClinicalBERT | 134.7 | 0.956 | 0.874 | 0.825 | 0.971 | 0.863 |
| 13 | google-bert/bert-base-uncased | 110.1 | 0.955 | 0.860 | 0.820 | 0.937 | 0.863 |
| 14 | microsoft/deberta-v3-base | 183.8 | 0.955 | 0.854 | 0.821 | 0.971 | 0.795 |
| 15 | distilbert/distilbert-base-uncased | 67.0 | 0.952 | 0.845 | 0.809 | 0.937 | 0.836 |
| 16 | google-bert/bert-base-cased | 108.9 | 0.948 | 0.832 | 0.797 | 0.931 | 0.849 |
| 17 | Simonlee711/Clinical_ModernBERT | 136.6 | 0.947 | 0.820 | 0.788 | 0.926 | 0.740 |
| 18 | microsoft/MiniLM-L12-H384-uncased | 33.4 | 0.931 | 0.590 | 0.682 | 0.943 | **0.000** |
| 19 | google/electra-base-discriminator | 108.9 | 0.932 | 0.579 | 0.707 | 0.989 | **0.000** |
| 20 | prajjwal1/bert-tiny | 4.4 | 0.925 | 0.558 | 0.693 | 0.966 | **0.000** |
| 21 | prajjwal1/bert-mini | 11.2 | 0.924 | 0.557 | 0.697 | 0.994 | **0.000** |
| | *Majority-class baseline* | | *0.877* | *0.311* | *0.000* | *0.000* | *0.000* |

Wilson confidence intervals, macro and weighted precision and recall, and quadratic weighted κ for every model are in `results/benchmark.csv`.

- **Accuracy is close to uninformative; macro-F1 separates the models.** Accuracy spans 6.6 points across the 21 encoders, of which the majority-class baseline already accounts for 0.877. Macro-F1 spans 0.557 to 0.964.
- **The four weakest models never predict weight loss.** Their recall on that class is exactly 0.000 while their accuracy stays between 0.924 and 0.932. They are two-class classifiers that have silently dropped the rarest category; no aggregate metric names that failure, per-class recall does.
- **Architecture generation beats scale.** Macro-F1 rises with parameter count (Spearman ρ = 0.79), but ModernBERT-base (150 M, 0.938) outperforms RoBERTa-large (355 M, 0.903) and sits 0.078 above BERT-base-uncased, a model of almost identical size.
- **Contrastive embedding backbones are efficient.** all-MiniLM-L6-v2, at 23 M parameters, reaches 0.874 macro-F1, within 0.064 of a 434 M model and above DeBERTa-v3-base, which has eight times as many parameters.
- **Biomedical pre-training does not help here.** The median macro-F1 of the three domain-specialised models (0.874) is slightly below that of the eighteen general-purpose ones (0.881). Patient reviews are written in lay register, not the formal idiom domain models are pre-trained on.
- **Models over-detect the minority classes.** Macro precision is below macro recall for 20 of the 21 models, a consequence of the class-weighted loss. For screening that is the preferable direction of error, but precision is the quantity to watch when building a downstream corpus.

## Repository layout

```
data/      empty; download the corpora from FigShare into it
scripts/   BERT_train.py         fine-tuning and evaluation
           fast_predictions.py   batched inference over a large table
results/   benchmark.csv         the 21 models, all metrics
           models_meta.csv       parameters, depth, context, family
CITATION.cff                    citation metadata
```

## Reproducing the benchmark

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 to 3.12. See the header of `requirements.txt` for CUDA variants.

### Fine-tune and evaluate

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

### Predict over a large file

```bash
python scripts/fast_predictions.py \
  -i reviews.parquet -o predictions.parquet \
  -m runs/modernbert-base --text-col REVIEW \
  --batch-size 128 --fp16 \
  --labels none gain loss --include-probabilities
```

Reads CSV, TSV, JSON, JSONL, Excel or Parquet. Use `--input-chunksize` (or `--auto-chunk`) for files that do not fit in memory; chunks are written to a temporary directory and merged at the end.

## Limitations

- **Not an external cohort.** The evaluation batch comes from the same four platforms, the same collection window and the same annotation team as the training corpus. Platform-held-out and temporally held-out evaluation remain open.
- **The target drug name is not part of the model input.** The dominant residual error is a review reporting weight change for a different drug than the one under review, and the model has no direct way to resolve that attribution. Appending the molecule to the input is the obvious next experiment.
- **No confidence interval on macro-F1.** Differences between adjacent models in the table should not be over-interpreted.
- **One shared protocol.** Models whose optimal learning rate differs markedly from the shared value are disadvantaged; the four weakest may be under-tuned rather than unsuitable.
- **Self-selected writers.** Reviews carry no verified demographic or clinical data, and platform content authenticity cannot be verified. Any downstream epidemiological use inherits that selection.

## Citation

If you use the code, the corpus or the benchmark, please cite the article and the dataset. The article reference will be completed on publication. The same metadata is in `CITATION.cff`, which feeds GitHub's **Cite this repository** button.

```bibtex
@article{yokoyama_weight_mentions,
  author  = {Yokoyama, Ta{\"i}oh and Natter, Johan and Godet, Julien},
  title   = {Detecting patient-reported mentions of weight change in online
             antidepressant reviews: an annotated corpus and a benchmark of
             21 transformer encoders},
  year    = {2026},
  note    = {Manuscript under review}
}

@misc{yokoyama_weight_mentions_data,
  author    = {Yokoyama, Ta{\"i}oh and Natter, Johan and Godet, Julien},
  title     = {Patient-reported weight-change mentions in online antidepressant
               reviews: two annotated corpora and an encoder benchmark},
  year      = {2026},
  publisher = {figshare},
  doi       = {10.6084/m9.figshare.33456934}
}
```

## Contributors and acknowledgements

**Authors.** Taïoh Yokoyama, Johan Natter and Julien Godet. TY and JG designed the study. TY and JN collected and curated the data and supervised annotation. JN ran the benchmark experiments. TY, JN and JG analysed the results and wrote the manuscript.

**Annotators.** The corpus exists because of the people who read and labelled the reviews. We thank Maxime Alter, Furkan Erol, Ahmed Guendouz, Vitoria Morais-Brazil, Paule Nkeng, Inas Oulkaid-Mouaddan, Sofia Salaa and Chinar Salmanli for their work on data annotation.

**Data.** We thank Askapatient.com for supplying the reviews covering rare molecules, monoamine oxidase inhibitors and several ATC classes that are sparsely represented on the other platforms.

## Funding and competing interests

This study did not receive any funding. The authors declare no competing interests.

## Licence

- **Code** (`scripts/`, `results/`): Apache License 2.0, see `LICENSE`.
- **Data**: Creative Commons Attribution 4.0 International (CC BY 4.0), distributed with the FigShare deposit rather than with this repository.

The data licence covers the annotations and the compilation of the corpora. The review texts were written by the patients who posted them and were collected from publicly accessible pages; that authorship is unaffected by the licence under which this work is released.

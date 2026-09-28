# Consumer Complaint Topic Modelling

**DLBDSEDA02 · Data Analysis · Wilhelmus Pretorius**

This project explores the recurring problems people describe in financial complaints. It prepares a large text dataset, compares four topic-modelling methods, and examines how clearly their topics describe the complaints.

The repository contains the **executed notebooks, PDF exports, result tables, graphs and an AI-assisted topic review**. You can inspect the completed work without downloading the dataset or running the models.

## Start here

| What you want to see | Where to look |
| --- | --- |
| How the texts were cleaned | [Preprocessing notebook](Data%20Preprocessing/complaints_preprocessing%20excecuted.ipynb) · [PDF](Data%20Preprocessing/complaints_preprocessing%20excecuted.pdf) |
| How the models were trained and compared | [Analysis notebook](Data%20Analysis/complaints_analysis%20executed.ipynb) · [PDF](Data%20Analysis/complaints_analysis%20executed.pdf) |
| Topic keywords, complaint assignments and comparison tables | [Result tables](Data%20Analysis%20Results/Excel%20Files) |
| Topic distributions, coverage, runtime and agreement | [Graphs](Data%20Analysis%20Results/Graphs) |
| Individual complaints checked against their assigned topics | [Topic-review workbook](Manual_complaint_topic_review.xlsx) |

*The folder named “Excel Files” contains CSV files, which can be opened in Excel or pandas.*

## Dataset and scope

The source is the [Consumer Complaint Database on Kaggle](https://www.kaggle.com/datasets/utkarshx27/consumer-complaint/data), uploaded by Utkarsh Singh. The downloaded file is `complaints.csv`.

The analysis uses the **Consumer complaint narrative** field. Complaint IDs, products and issues are retained for identification and inspection; product and issue labels are not supplied to the topic models.

| Stage | Complaints |
| --- | ---: |
| Rows read from the original CSV | 3,585,952 |
| Narratives retained after preprocessing | 1,292,206 |
| Training rows used by each model | 1,200,000 |
| Separate held-out rows used for comparison | 1,000 |

These figures describe the saved run. Most removed rows had missing or blank narratives. The graphs and assignment CSVs show the **1,000 held-out complaints**, while the models were trained on **1.2 million**. Assignment of the entire remaining corpus is disabled in the executed analysis.

The raw CSV, generated preprocessing data, embedding caches and fitted models are not included in this repository. Download the data separately to reproduce the workflow.

## How the workflow works

1. **Prepare the text.** Read small pandas batches, discard blank narratives, check repeated complaint IDs, normalise text, remove HTML, URLs, email addresses and redaction placeholders, then tokenise and lemmatise with NLTK. Save records as JSONL and word features as sparse matrices to control RAM use.
2. **Build analysis features.** Read the saved JSONL, select training and held-out complaints, expand contractions, and rebuild word-count and TF-IDF features. Common English stop words are removed from word features while `no`, `not`, `nor`, `never` and `without` are retained. Sentence embeddings use contraction-expanded, lightly cleaned original text.
3. **Discover and compare topics.** Fit LSA, LDA, NMF and a BERTopic variant, assign the held-out complaints, and save topic words, assignments, comparisons and graphs.
4. **Inspect examples.** Read selected complaints alongside their topic keywords to assess whether the topics make sense.

**No stemming is used.** The preprocessing notebook retains stop words; their removal happens when the analysis notebook builds its word features.

## The four methods

| Method | Input | What it does in this project |
| --- | --- | --- |
| **LSA — Latent Semantic Analysis** | TF-IDF | Uses truncated singular value decomposition to find broad word patterns. Six components are interpreted through their positive and negative poles, giving up to 12 theme labels. |
| **LDA — Latent Dirichlet Allocation** | Word counts | Estimates each complaint as a mixture of topics. This project uses a custom PyTorch CUDA implementation with 12 topics. |
| **NMF — Non-negative Matrix Factorisation** | TF-IDF | Builds topics from additive groups of weighted words using mini-batch NMF with 12 components. |
| **BERTopic variant** | MiniLM sentence embeddings | Groups semantically similar complaints into 12 MiniBatchKMeans clusters and describes them with class-based TF-IDF. |

The BERTopic implementation uses `sentence-transformers/all-MiniLM-L6-v2`, batched GPU embedding and disk-backed storage. It does **not** use the usual UMAP/HDBSCAN pipeline or detect outliers. GPU use therefore does not mean every processing stage runs on the GPU.

## Results at a glance

Topics include credit-reporting errors, identity theft, debt collection, card charges, mortgages, student loans and late-payment disputes.

![Topic assignments across the 1,000 held-out complaints](Data%20Analysis%20Results/Graphs/topic_sizes.png)

All four models assigned the 1,000 held-out complaints. LSA used nine signed component labels; the other models used all 12 topics or clusters. **Assignment coverage is not prediction accuracy.**

The highest pairwise Adjusted Rand Index was between LDA and BERTopic, at approximately **0.283**. This measures agreement between their groupings, not whether either grouping is correct.

- [Model comparison](Data%20Analysis%20Results/Excel%20Files/model_comparison.csv) — training/evaluation sizes, coverage and recorded runtime.
- [Pairwise agreement](Data%20Analysis%20Results/Excel%20Files/pairwise_agreement.csv) — ARI and adjusted mutual information on shared complaints.
- [Representative complaints](Data%20Analysis%20Results/Excel%20Files/representative_complaints.csv) — selected examples for interpretation.
- [Agreement heatmap](Data%20Analysis%20Results/Graphs/agreement_ari.png) · [Coverage and runtime graph](Data%20Analysis%20Results/Graphs/coverage_and_runtime.png).

The topic-review workbook contains **AI-assisted judgements**, not independent human validation. The selected examples are not a random, equally sampled test across models, so their fit counts should not be used as model accuracy or a ranking. Repeated complaint templates and broad, overlapping topics also affect interpretation. Recorded runtimes include different cache/resume effects and are not a controlled speed benchmark.

## Running the notebooks

### 1. Set up Python

Use **Python 3.10**, Conda and Jupyter. The full run was developed on a laptop with 32 GB RAM and an NVIDIA RTX 4070. The current LDA and BERTopic cells require a CUDA-capable NVIDIA GPU and a CUDA-enabled PyTorch installation.

```bash
conda create -n complaints python=3.10 -y
conda activate complaints
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name complaints --display-name "Python (complaints)"
jupyter lab
```

Select the **Python (complaints)** kernel. Install the dependency list shown near the top of the analysis notebook in a temporary cell, then restart the kernel. It covers NumPy, pandas, SciPy, scikit-learn, NLTK, joblib, psutil, Matplotlib, seaborn, PyTorch, SentenceTransformers, BERTopic and notebook-export packages.

Use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to select a compatible CUDA build, respecting the notebook's version constraints. Before running the GPU cells, check:

```python
import torch
print(torch.cuda.is_available())  # Must be True for the current GPU cells.
```

The first run downloads NLTK resources and the sentence-embedding model. Leave additional disk space for generated text, sparse features, checkpoints and embedding caches.

### 2. Run preprocessing

Open the [preprocessing notebook](Data%20Preprocessing/complaints_preprocessing%20excecuted.ipynb), change `DATA_PATH` to your downloaded CSV, and run the processing cells in order.

```python
from pathlib import Path
DATA_PATH = Path(r"C:\your\data\complaints.csv")
```

Keep `KEEP_ORIGINAL_TEXT = True` because the analysis needs the original narratives. `MAX_SOURCE_ROWS = None` processes the complete CSV; a smaller integer is useful for a first test. Completed results are written beneath `preprocessing_streamed` beside the CSV.

### 3. Run analysis

Open the [analysis notebook](Data%20Analysis/complaints_analysis%20executed.ipynb) in a fresh kernel. Set `PREP_ROOT` to the generated `preprocessing_streamed` folder and run the analysis cells in order. `PREP_RUN_DIR = None` selects the newest completed preprocessing run.

New results are saved under `PREP_ROOT.parent / "analysis_results" / "run_..."`, including tables, figures, model files and run metadata.

The recorded settings use `TRAIN_DOCS = 1_200_000` and `EVAL_DOCS = 1_000`. Reduce the training size for an initial test. Start a fresh analysis run after changing text-processing or model settings, rather than reusing incompatible checkpoints.

### Notes before rerunning

- Paths in the uploaded notebooks point to the original Windows machine and must be changed locally.
- Some explanatory text still mentions older defaults, including 100,000 training rows and retained stop words. The current code and saved result tables reflect the settings described above.
- PDF export is optional; ready-made PDFs are linked at the top. Set `NOTEBOOK_PATH` to the actual local filename and save the notebook first. In the analysis notebook, edit this path **inside the final export cell**, which overrides the earlier setting. That cell also needs its Windows `OUT_DIR` string changed to a raw string, such as `Path(r"C:\your\exports")`, and the output directory created before writing.

---

Created by **Wilhelmus Pretorius** for the **DLBDSEDA02 Data Analysis** project. The results describe themes in submitted complaints, not the prevalence of problems across all customers.

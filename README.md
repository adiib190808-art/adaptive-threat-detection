# Adaptive Multi-Threat Cybersecurity Text Detector

An adaptive machine-learning system for detecting suspicious text and classifying detected threats into cybersecurity threat families.

The project uses a two-stage NLP architecture:

1. **Stage 1 — Threat Detection:** Determines whether input text is SAFE or THREAT.
2. **Stage 2 — Threat Classification:** If a threat is detected, assigns it to a cybersecurity threat family.

The system also contains a human-feedback mechanism that allows incorrect predictions to be corrected and incorporated into future model retraining.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │     Input Text      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Character TF-IDF    │
                         │    3–5 grams        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Stage 1 LinearSVC   │
                         │   SAFE / THREAT     │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                     SAFE                      THREAT
                       │                         │
                       ▼                         ▼
                  Final Result          ┌─────────────────────┐
                                        │ Character TF-IDF    │
                                        │    3–5 grams        │
                                        └──────────┬──────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │ Stage 2 LinearSVC   │
                                        │ Threat Family       │
                                        └──────────┬──────────┘
                                                   │
                                                   ▼
                                              Final Result
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │ Human Feedback      │
                                        └──────────┬──────────┘
                                                   │
                                                   ▼
                                        Adaptive Corrections
                                                   │
                                                   ▼
                                         Weighted Retraining
```

---

## Threat Families

Stage 2 can classify detected threats into eight broad families:

- Social Engineering
- Prompt Injection
- Phishing & Credential Theft
- Data / Secret Exfiltration
- Fraud & Financial Scam
- Impersonation
- System & Tool Abuse
- Other Suspicious

These are broad project-defined families created by mapping the more granular scenario categories in the training corpus.

---

## Machine Learning

### Stage 1 — SAFE vs THREAT

Feature extraction:

```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=300000
)
```

Classifier:

```python
LinearSVC(C=0.25)
```

The Stage 1 configuration was selected using validation data with emphasis on strong threat recall while maintaining overall F1 performance.

### Stage 2 — Threat Family

Stage 2 uses character-level TF-IDF with the same general feature configuration.

Classifier:

```python
LinearSVC(
    C=1,
    class_weight="balanced"
)
```

Class weighting is used because the threat-family distribution is highly imbalanced.

---

## Adaptive Human Feedback

The detector can ask the user whether a prediction was correct.

If a prediction is wrong, the user can provide the correct SAFE/THREAT label and, when applicable, the correct threat family.

Corrections are stored in:

```text
adaptive_corrections.csv
```

During the next training cycle, human corrections receive a larger sample weight:

```python
ADAPTIVE_WEIGHT = 20
```

The current implementation uses **controlled batch retraining** rather than automatically changing the model after every individual input.

This reduces the risk of uncontrolled model changes while still allowing the system to adapt to trusted human feedback.

---

## Demonstrated Adaptation

During testing, a phishing-style account-verification email was correctly detected as a threat but initially assigned to the wrong threat family.

### Before feedback

```text
Status: THREAT
Threat family: Data / Secret Exfiltration
Stage 1 decision score: 1.1015
Stage 2 decision score: 0.6429
```

The user corrected the threat family to:

```text
Phishing & Credential Theft
```

The correction was stored and included with increased weight during retraining.

### After retraining

```text
Status: THREAT
Threat family: Phishing & Credential Theft
Stage 1 decision score: 1.1017
Stage 2 decision score: 0.9715
```

This demonstrates the complete adaptive cycle:

```text
Prediction
    ↓
Human Feedback
    ↓
Correction Storage
    ↓
Weighted Retraining
    ↓
Updated Prediction
```

Decision-function values shown by the application are **LinearSVC decision scores, not calibrated probabilities**.

---

## Dataset

The main multi-threat classifier was trained using the **ScamBench Training Corpus**, a multilingual and multi-scenario dataset containing approximately 37,000 examples across many scam and adversarial-interaction categories.

Only user-generated message content is extracted for model input.

Fields that could directly expose target labels or reasoning are deliberately excluded from model features, including fields such as:

- decision labels
- explanations
- reasoning traces
- unsafe-signal annotations
- diagnostic labels
- chosen actions

This is intended to reduce direct target leakage.

The project also began with experiments on phishing/spam email classification before expanding into broader multi-threat text detection.

### Dataset sources

ScamBench Training Corpus:

https://huggingface.co/datasets/shaw/scambench-training

Phishing Email Dataset:

https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data

The phishing email dataset combines several established email corpora. Its positive class represents a broader spam/phishing category rather than exclusively phishing.

---

## Final Evaluation

Hyperparameters were selected using validation data before the final test evaluation.

### Leakage-Clean Test

An additional test subset was created by removing test rows whose extracted text exactly matched text appearing in the training set.

The leakage-clean subset contained **3,385 examples**.

### Stage 1 — Threat Detection

| Metric | Result |
|---|---:|
| Accuracy | 87.15% |
| Precision | 82.56% |
| Recall | **94.98%** |
| F1 | **88.33%** |
| Macro F1 | 87.02% |

Confusion matrix:

```text
[[1303, 348],
 [  87, 1647]]
```

Stage 1 therefore detected approximately **94.98% of threat examples** in the leakage-clean evaluation.

### Stage 2 — Threat-Family Classification

Evaluated on ground-truth threat examples in the leakage-clean test subset:

| Metric | Result |
|---|---:|
| Accuracy | **94.52%** |
| Balanced Accuracy | 84.47% |
| Macro F1 | 84.57% |
| Weighted F1 | 94.48% |

### Complete Two-Stage System

| Metric | Result |
|---|---:|
| Accuracy | **84.37%** |
| Macro F1 | **81.15%** |
| Weighted F1 | 84.44% |

The end-to-end evaluation includes both Stage 1 gating and Stage 2 threat-family classification.

---

## Why Character N-Grams?

Character-level TF-IDF was chosen because cybersecurity text often contains:

- unusual URLs
- modified spellings
- obfuscated words
- suspicious fragments
- multilingual text
- punctuation variations
- credential-related patterns

Character n-grams can capture many of these patterns without relying entirely on exact whole-word vocabulary.

---

## Running the Project

### 1. Clone the repository

git clone https://github.com/adiib190808-art/adaptive-threat-detection.git
cd adaptive-threat-detection

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Obtain the dataset

Download the required ScamBench parquet training data and place the training file in the project directory as:

```text
train-00000-of-00001.parquet
```

Large dataset files are intentionally excluded from Git using `.gitignore`.

### 4. Train the models

```bash
python train_models.py
```

This creates:

```text
models/
├── stage1_model.joblib
├── stage1_vectorizer.joblib
├── stage2_model.joblib
└── stage2_vectorizer.joblib
```

### 5. Run the detector

```bash
python detector.py
```

Paste or type a message and enter:

```text
END
```

on a new line when the message is complete.

The detector will return either:

```text
SAFE
```

or:

```text
THREAT
Threat family: <predicted family>
```

The user can then provide feedback on the prediction.

If corrections have been added, run `train_models.py` again to incorporate them into the next model version.

---

## Project Files

```text
cyber-threat-detector/
│
├── detector.py
├── train_models.py
├── adaptive_feedback.py
├── adaptive_corrections.csv
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    ├── stage1_model.joblib
    ├── stage1_vectorizer.joblib
    ├── stage2_model.joblib
    └── stage2_vectorizer.joblib
```

The generated model files and large parquet datasets are excluded from Git by default.

---

## Limitations

This project is an **adaptive and extensible multi-threat text-classification prototype**, not a production cybersecurity system.

Important limitations include:

- ScamBench contains synthetic and multi-source adversarial conversations and is not purely a real-world email corpus.
- Exact text overlap was found between the official dataset splits.
- The leakage-clean evaluation removes exact train/test matches, but does not guarantee removal of near-duplicate, template, or source-level leakage.
- Several threat families have relatively small evaluation support.
- Some cybersecurity categories overlap semantically.
- Stage 1 intentionally favors high threat recall, which can increase false positives.
- LinearSVC decision scores are not probabilities.
- Human corrections can reflect organizational or user policy rather than universal ground truth.
- The adaptive mechanism assumes feedback comes from a trusted human.
- The system analyzes text and does not inspect attachments, URLs, network behavior, sender authentication, or executable content.

The system should therefore be treated as a research/learning prototype rather than a replacement for professional security infrastructure.

---

## Future Improvements

Possible extensions include:

- calibrated confidence estimates
- URL-specific feature analysis
- attachment metadata analysis
- email-header and sender-authentication features
- multilingual transformer models
- near-duplicate detection during dataset splitting
- stronger external validation datasets
- correction review/approval workflows
- model-version tracking
- adversarial robustness testing
- specialized models for individual threat families

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- LinearSVC
- joblib
- PyArrow

---

## Purpose

This project was built as a practical exploration of machine learning, natural-language processing, cybersecurity classification, model evaluation, error analysis, data leakage, and human-guided adaptive learning.

Rather than focusing only on benchmark accuracy, the project explores how a text-classification system can identify its mistakes, receive trusted human corrections, and incorporate those corrections into future model versions.
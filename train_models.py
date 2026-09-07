import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_FILE = "train-00000-of-00001.parquet"
CORRECTION_FILE = "adaptive_corrections.csv"

MODEL_FOLDER = "models"

ADAPTIVE_WEIGHT = 20


# =========================================================
# MODEL OUTPUT FILES
# =========================================================

STAGE1_MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "stage1_model.joblib"
)

STAGE1_VECTORIZER_FILE = os.path.join(
    MODEL_FOLDER,
    "stage1_vectorizer.joblib"
)

STAGE2_MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "stage2_model.joblib"
)

STAGE2_VECTORIZER_FILE = os.path.join(
    MODEL_FOLDER,
    "stage2_vectorizer.joblib"
)


# =========================================================
# EXTRACT USER TEXT
# =========================================================

def extract_user_text(messages):

    conversation = json.loads(messages)

    user_messages = []

    for message in conversation:

        if message.get("role") == "user":

            content = message.get(
                "content",
                ""
            )

            if content is not None:

                user_messages.append(
                    str(content)
                )

    return " ".join(
        user_messages
    )


# =========================================================
# THREAT FAMILY MAPPING
# =========================================================

def map_threat_family(category):

    category = str(category).lower()


    if category == "borderline-suspicious":
        return "Other Suspicious"


    prompt_terms = [
        "prompt-injection",
        "prompt injection",
        "jailbreak",
        "instruction-hijack",
        "instruction hijack",
        "system-prompt",
        "system prompt"
    ]

    if any(
        term in category
        for term in prompt_terms
    ):
        return "Prompt Injection"


    exfiltration_terms = [
        "secret-exfiltration",
        "secret exfiltration",
        "data-exfiltration",
        "data exfiltration",
        "private-data",
        "private data",
        "sensitive-data",
        "sensitive data",
        "api-key",
        "api key",
        "environment-variable",
        "environment variable"
    ]

    if any(
        term in category
        for term in exfiltration_terms
    ):
        return "Data / Secret Exfiltration"


    phishing_terms = [
        "phishing",
        "credential",
        "password",
        "login",
        "account-takeover",
        "account takeover",
        "verification-code",
        "verification code",
        "2fa",
        "seed-phrase",
        "seed phrase",
        "recovery-phrase",
        "recovery phrase"
    ]

    if any(
        term in category
        for term in phishing_terms
    ):
        return "Phishing & Credential Theft"


    fraud_terms = [
        "fraud",
        "advance-fee",
        "advance fee",
        "investment",
        "crypto",
        "financial",
        "payment",
        "money",
        "lottery",
        "prize",
        "romance-scam",
        "romance scam"
    ]

    if any(
        term in category
        for term in fraud_terms
    ):
        return "Fraud & Financial Scam"


    impersonation_terms = [
        "impersonation",
        "identity-spoof",
        "identity spoof",
        "fake-support",
        "fake support"
    ]

    if any(
        term in category
        for term in impersonation_terms
    ):
        return "Impersonation"


    system_terms = [
        "malicious-tool",
        "malicious tool",
        "tool-abuse",
        "tool abuse",
        "command-execution",
        "command execution",
        "system-abuse",
        "system abuse"
    ]

    if any(
        term in category
        for term in system_terms
    ):
        return "System & Tool Abuse"


    return "Social Engineering"


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("=" * 65)
print("TRAINING FINAL CYBERSECURITY MODELS")
print("=" * 65)


print("\nLoading training data...")


train_df = pd.read_parquet(
    TRAIN_FILE
)


train_df["text"] = (
    train_df["messages"]
    .apply(extract_user_text)
)


print(
    "Training rows:",
    len(train_df)
)


# =========================================================
# LOAD ADAPTIVE CORRECTIONS
# =========================================================

if os.path.exists(
    CORRECTION_FILE
):

    corrections_df = pd.read_csv(
        CORRECTION_FILE
    )

    corrections_df = corrections_df.drop_duplicates(
        subset=["text"],
        keep="last"
    )

    print(
        "Adaptive corrections loaded:",
        len(corrections_df)
    )

else:

    corrections_df = pd.DataFrame(
        columns=[
            "text",
            "is_threat",
            "threat_family"
        ]
    )

    print(
        "No adaptive corrections found."
    )


# =========================================================
# CREATE MODEL FOLDER
# =========================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# =========================================================
# STAGE 1 TRAINING DATA
# =========================================================

print("\n")
print("=" * 65)
print("PREPARING STAGE 1")
print("=" * 65)


stage1_text = list(
    train_df["text"]
)


stage1_labels = list(
    train_df[
        "should_trigger_scam_defense"
    ]
    .astype(int)
)


stage1_weights = [
    1.0
] * len(stage1_text)


for _, correction in corrections_df.iterrows():

    stage1_text.append(
        str(
            correction["text"]
        )
    )

    stage1_labels.append(
        int(
            correction["is_threat"]
        )
    )

    stage1_weights.append(
        float(
            ADAPTIVE_WEIGHT
        )
    )


stage1_labels = np.array(
    stage1_labels
)

stage1_weights = np.array(
    stage1_weights
)


print(
    "Stage 1 training examples:",
    len(stage1_text)
)


# =========================================================
# STAGE 1 VECTORIZER
#
# FINAL VALIDATION-SELECTED CONFIGURATION
# =========================================================

print(
    "\nBuilding Stage 1 TF-IDF features..."
)


stage1_vectorizer = TfidfVectorizer(

    analyzer="char_wb",

    ngram_range=(3, 5),

    min_df=2,

    sublinear_tf=True,

    max_features=300000
)


X_stage1 = (
    stage1_vectorizer
    .fit_transform(
        stage1_text
    )
)


print(
    "Stage 1 feature shape:",
    X_stage1.shape
)


# =========================================================
# STAGE 1 MODEL
#
# FINAL FROZEN CONFIGURATION
# C = 0.25
# =========================================================

print(
    "\nTraining Stage 1..."
)


stage1_model = LinearSVC(
    C=0.25
)


stage1_model.fit(
    X_stage1,
    stage1_labels,
    sample_weight=stage1_weights
)


print(
    "Stage 1 training complete."
)


# =========================================================
# STAGE 2 TRAINING DATA
# =========================================================

print("\n")
print("=" * 65)
print("PREPARING STAGE 2")
print("=" * 65)


threat_train_df = train_df[
    train_df[
        "should_trigger_scam_defense"
    ] == True
].copy()


threat_train_df["family"] = (
    threat_train_df[
        "scenario_category"
    ]
    .apply(map_threat_family)
)


stage2_text = list(
    threat_train_df["text"]
)


stage2_labels = list(
    threat_train_df["family"]
)


stage2_weights = [
    1.0
] * len(stage2_text)


# =========================================================
# ADD THREAT CORRECTIONS TO STAGE 2
# =========================================================

for _, correction in corrections_df.iterrows():

    if int(
        correction["is_threat"]
    ) == 1:

        family = correction[
            "threat_family"
        ]

        if pd.notna(family):

            stage2_text.append(
                str(
                    correction["text"]
                )
            )

            stage2_labels.append(
                str(family)
            )

            stage2_weights.append(
                float(
                    ADAPTIVE_WEIGHT
                )
            )


stage2_labels = np.array(
    stage2_labels
)

stage2_weights = np.array(
    stage2_weights
)


print(
    "Stage 2 training examples:",
    len(stage2_text)
)


print(
    "\nThreat family distribution:"
)


print(
    pd.Series(
        stage2_labels
    )
    .value_counts()
)


# =========================================================
# STAGE 2 VECTORIZER
# =========================================================

print(
    "\nBuilding Stage 2 TF-IDF features..."
)


stage2_vectorizer = TfidfVectorizer(

    analyzer="char_wb",

    ngram_range=(3, 5),

    min_df=2,

    sublinear_tf=True,

    max_features=300000
)


X_stage2 = (
    stage2_vectorizer
    .fit_transform(
        stage2_text
    )
)


print(
    "Stage 2 feature shape:",
    X_stage2.shape
)


# =========================================================
# STAGE 2 MODEL
#
# FINAL FROZEN CONFIGURATION
# =========================================================

print(
    "\nTraining Stage 2..."
)


stage2_model = LinearSVC(
    C=1,
    class_weight="balanced"
)


stage2_model.fit(
    X_stage2,
    stage2_labels,
    sample_weight=stage2_weights
)


print(
    "Stage 2 training complete."
)


# =========================================================
# SAVE MODELS
# =========================================================

print("\n")
print("=" * 65)
print("SAVING MODELS")
print("=" * 65)


joblib.dump(
    stage1_model,
    STAGE1_MODEL_FILE
)


joblib.dump(
    stage1_vectorizer,
    STAGE1_VECTORIZER_FILE
)


joblib.dump(
    stage2_model,
    STAGE2_MODEL_FILE
)


joblib.dump(
    stage2_vectorizer,
    STAGE2_VECTORIZER_FILE
)


print(
    "Saved:",
    STAGE1_MODEL_FILE
)

print(
    "Saved:",
    STAGE1_VECTORIZER_FILE
)

print(
    "Saved:",
    STAGE2_MODEL_FILE
)

print(
    "Saved:",
    STAGE2_VECTORIZER_FILE
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)


print(
    "Stage 1:"
)

print(
    "Character TF-IDF (3-5 grams)"
)

print(
    "LinearSVC C=0.25"
)


print(
    "\nStage 2:"
)

print(
    "Character TF-IDF (3-5 grams)"
)

print(
    "LinearSVC C=1"
)

print(
    "class_weight='balanced'"
)


print(
    "\nAdaptive correction weight:",
    ADAPTIVE_WEIGHT
)


print(
    "\nAll models saved successfully."
)

print(
    "\nYou can now run detector.py"
)
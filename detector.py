import os
import joblib
import numpy as np

from adaptive_feedback import collect_feedback


# =========================================================
# MODEL PATHS
# =========================================================

MODEL_FOLDER = "models"

STAGE1_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "stage1_model.joblib"
)

STAGE1_VECTORIZER_PATH = os.path.join(
    MODEL_FOLDER,
    "stage1_vectorizer.joblib"
)

STAGE2_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "stage2_model.joblib"
)

STAGE2_VECTORIZER_PATH = os.path.join(
    MODEL_FOLDER,
    "stage2_vectorizer.joblib"
)


# =========================================================
# LOAD MODELS
# =========================================================

def load_models():

    required_files = [
        STAGE1_MODEL_PATH,
        STAGE1_VECTORIZER_PATH,
        STAGE2_MODEL_PATH,
        STAGE2_VECTORIZER_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Missing model file: {file_path}\n"
                "Run train_models.py first."
            )

    stage1_model = joblib.load(
        STAGE1_MODEL_PATH
    )

    stage1_vectorizer = joblib.load(
        STAGE1_VECTORIZER_PATH
    )

    stage2_model = joblib.load(
        STAGE2_MODEL_PATH
    )

    stage2_vectorizer = joblib.load(
        STAGE2_VECTORIZER_PATH
    )

    return (
        stage1_model,
        stage1_vectorizer,
        stage2_model,
        stage2_vectorizer
    )


# =========================================================
# MULTI-LINE MESSAGE INPUT
# =========================================================

def get_message():

    print("\nPaste or type the message below.")
    print("Type END on a new line when finished.")
    print("Type QUIT on the first line to close the detector.\n")

    lines = []

    while True:

        line = input()

        # Allow quitting before entering a message
        if not lines and line.strip().lower() in [
            "quit",
            "exit",
            "q"
        ]:
            return None

        if line.strip().upper() == "END":
            break

        lines.append(line)

    text = "\n".join(lines).strip()

    return text


# =========================================================
# ANALYZE MESSAGE
# =========================================================

def analyze_message(
    text,
    stage1_model,
    stage1_vectorizer,
    stage2_model,
    stage2_vectorizer
):

    # -----------------------------------------------------
    # Stage 1: SAFE vs THREAT
    # -----------------------------------------------------

    X_stage1 = stage1_vectorizer.transform(
        [text]
    )

    stage1_prediction = int(
        stage1_model.predict(
            X_stage1
        )[0]
    )

    stage1_score = float(
        stage1_model.decision_function(
            X_stage1
        )[0]
    )

    # SAFE
    if stage1_prediction == 0:

        return {
            "status": "SAFE",
            "threat_family": None,
            "stage1_score": stage1_score,
            "stage2_score": None
        }

    # -----------------------------------------------------
    # Stage 2: Threat-family classification
    # -----------------------------------------------------

    X_stage2 = stage2_vectorizer.transform(
        [text]
    )

    family_prediction = stage2_model.predict(
        X_stage2
    )[0]

    family_scores = stage2_model.decision_function(
        X_stage2
    )[0]

    winning_index = int(
        np.argmax(family_scores)
    )

    winning_score = float(
        family_scores[winning_index]
    )

    return {
        "status": "THREAT",
        "threat_family": str(family_prediction),
        "stage1_score": stage1_score,
        "stage2_score": winning_score
    }


# =========================================================
# DISPLAY RESULT
# =========================================================

def display_result(result):

    print("\n")
    print("=" * 55)
    print("ANALYSIS RESULT")
    print("=" * 55)

    print(
        f"Status: {result['status']}"
    )

    if result["status"] == "THREAT":

        print(
            "Threat family:",
            result["threat_family"]
        )

    print(
        "Stage 1 decision score:",
        f"{result['stage1_score']:.4f}"
    )

    if result["stage2_score"] is not None:

        print(
            "Stage 2 decision score:",
            f"{result['stage2_score']:.4f}"
        )

    print("-" * 55)

    if result["status"] == "SAFE":

        print(
            "Interpretation: "
            "The model classified this message as safe."
        )

    else:

        print(
            "Interpretation: "
            "The model detected suspicious content."
        )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    print("=" * 60)
    print("ADAPTIVE MULTI-THREAT CYBERSECURITY DETECTOR")
    print("=" * 60)

    print("\nLoading trained models...")

    (
        stage1_model,
        stage1_vectorizer,
        stage2_model,
        stage2_vectorizer
    ) = load_models()

    print("Models loaded successfully.")

    while True:

        text = get_message()

        if text is None:

            print(
                "\nCybersecurity detector closed."
            )
            break

        if not text:

            print(
                "\nPlease enter a message."
            )
            continue

        result = analyze_message(
            text,
            stage1_model,
            stage1_vectorizer,
            stage2_model,
            stage2_vectorizer
        )

        display_result(
            result
        )

        # -------------------------------------------------
        # Adaptive human feedback
        # -------------------------------------------------

        collect_feedback(
            text=text,
            predicted_status=result["status"],
            predicted_family=result["threat_family"]
        )


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":
    main()
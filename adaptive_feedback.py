import os
import pandas as pd


CORRECTION_FILE = "adaptive_corrections.csv"

THREAT_FAMILIES = [
    "Social Engineering",
    "Prompt Injection",
    "Phishing & Credential Theft",
    "Data / Secret Exfiltration",
    "Fraud & Financial Scam",
    "Impersonation",
    "System & Tool Abuse",
    "Other Suspicious"
]


def save_correction(text, is_threat, threat_family=None):

    new_correction = pd.DataFrame([
        {
            "text": text,
            "is_threat": int(is_threat),
            "threat_family": threat_family
        }
    ])

    if os.path.exists(CORRECTION_FILE):

        corrections = pd.read_csv(CORRECTION_FILE)

        corrections = pd.concat(
            [corrections, new_correction],
            ignore_index=True
        )

    else:

        corrections = new_correction

    # If the same message was corrected more than once,
    # keep the newest correction.
    corrections = corrections.drop_duplicates(
        subset=["text"],
        keep="last"
    )

    corrections.to_csv(
        CORRECTION_FILE,
        index=False
    )

    print("\nCorrection saved successfully.")
    print("It will be used during the next model retraining.")


def get_threat_family():

    print("\nChoose the correct threat family:\n")

    for number, family in enumerate(
        THREAT_FAMILIES,
        start=1
    ):
        print(f"{number}. {family}")

    while True:

        choice = input("\nEnter number: ").strip()

        try:
            choice = int(choice)

            if 1 <= choice <= len(THREAT_FAMILIES):
                return THREAT_FAMILIES[choice - 1]

        except ValueError:
            pass

        print("Invalid choice. Try again.")


def collect_feedback(text, predicted_status, predicted_family=None):

    print("\nWas this prediction correct?")
    answer = input("Enter y/n: ").strip().lower()

    if answer in ["y", "yes"]:
        print("No correction needed.")
        return

    if answer not in ["n", "no"]:
        print("Feedback skipped.")
        return

    print("\nWhat should the correct status be?")
    print("1. SAFE")
    print("2. THREAT")

    while True:

        choice = input("\nEnter number: ").strip()

        if choice == "1":

            save_correction(
                text=text,
                is_threat=0,
                threat_family=None
            )

            return

        elif choice == "2":

            family = get_threat_family()

            save_correction(
                text=text,
                is_threat=1,
                threat_family=family
            )

            return

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":

    print("Adaptive Feedback Module")
    print("This module is designed to be used by detector.py.")
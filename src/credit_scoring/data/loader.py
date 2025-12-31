"""Data loader for German Credit Dataset.

This module handles loading and basic preprocessing of the German Credit
dataset from UCI Machine Learning Repository.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_german_credit(
    save_raw: bool = True,
    data_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the German Credit Dataset from UCI Repository.

    The German Credit dataset contains 1000 instances with 20 features
    describing credit applicants. The target variable indicates whether
    the applicant is a good (1) or bad (2) credit risk.

    Args:
        save_raw: Whether to save the raw data to disk.
        data_dir: Directory to save raw data. Defaults to project's data/raw.

    Returns:
        Tuple of (features DataFrame, target Series).

    Example:
        >>> X, y = load_german_credit()
        >>> print(f"Features shape: {X.shape}")
        >>> print(f"Target distribution: {y.value_counts()}")
    """
    # Fetch dataset from UCI repository
    german_credit = fetch_ucirepo(id=144)

    # Extract features and target
    X = german_credit.data.features
    y = german_credit.data.targets

    # Flatten target if it's a DataFrame
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Convert target to binary (0 = good, 1 = bad/default)
    # Original: 1 = good, 2 = bad
    y = (y == 2).astype(int)
    y.name = "default"

    # Save raw data if requested
    if save_raw:
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Combine features and target for saving
        df = X.copy()
        df["default"] = y
        df.to_csv(data_dir / "german_credit.csv", index=False)
        print(f"[OK] Raw data saved to: {data_dir / 'german_credit.csv'}")

    return X, y


def load_from_csv(
    filepath: Path | str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load German Credit data from local CSV file.

    Args:
        filepath: Path to CSV file. Defaults to data/raw/german_credit.csv.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    if filepath is None:
        filepath = Path(__file__).parents[3] / "data" / "raw" / "german_credit.csv"

    df = pd.read_csv(filepath)
    y = df.pop("default")
    X = df

    return X, y


def get_feature_info() -> dict:
    """Get metadata about the German Credit Dataset features.

    Returns:
        Dictionary with feature descriptions and types.
    """
    return {
        "Attribute1": {
            "name": "checking_status",
            "description": "Status of existing checking account",
            "type": "categorical",
        },
        "Attribute2": {
            "name": "duration",
            "description": "Duration in months",
            "type": "numerical",
        },
        "Attribute3": {
            "name": "credit_history",
            "description": "Credit history",
            "type": "categorical",
        },
        "Attribute4": {
            "name": "purpose",
            "description": "Purpose of the credit",
            "type": "categorical",
        },
        "Attribute5": {
            "name": "credit_amount",
            "description": "Credit amount",
            "type": "numerical",
        },
        "Attribute6": {
            "name": "savings_status",
            "description": "Savings account/bonds",
            "type": "categorical",
        },
        "Attribute7": {
            "name": "employment",
            "description": "Present employment since",
            "type": "categorical",
        },
        "Attribute8": {
            "name": "installment_commitment",
            "description": "Installment rate (% of disposable income)",
            "type": "numerical",
        },
        "Attribute9": {
            "name": "personal_status",
            "description": "Personal status and sex",
            "type": "categorical",
        },
        "Attribute10": {
            "name": "other_parties",
            "description": "Other debtors/guarantors",
            "type": "categorical",
        },
        "Attribute11": {
            "name": "residence_since",
            "description": "Present residence since",
            "type": "numerical",
        },
        "Attribute12": {
            "name": "property_magnitude",
            "description": "Property type",
            "type": "categorical",
        },
        "Attribute13": {
            "name": "age",
            "description": "Age in years",
            "type": "numerical",
        },
        "Attribute14": {
            "name": "other_payment_plans",
            "description": "Other installment plans",
            "type": "categorical",
        },
        "Attribute15": {
            "name": "housing",
            "description": "Housing type",
            "type": "categorical",
        },
        "Attribute16": {
            "name": "existing_credits",
            "description": "Number of existing credits at this bank",
            "type": "numerical",
        },
        "Attribute17": {
            "name": "job",
            "description": "Job type",
            "type": "categorical",
        },
        "Attribute18": {
            "name": "num_dependents",
            "description": "Number of people being liable for",
            "type": "numerical",
        },
        "Attribute19": {
            "name": "own_telephone",
            "description": "Telephone registered under customer name",
            "type": "binary",
        },
        "Attribute20": {
            "name": "foreign_worker",
            "description": "Is foreign worker",
            "type": "binary",
        },
    }


if __name__ == "__main__":
    # Quick test
    print("Loading German Credit Dataset...")
    X, y = load_german_credit()
    print(f"\n[OK] Dataset loaded successfully!")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"\n[*] Target distribution:")
    print(f"   Good credit (0): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"   Bad credit (1): {(y == 1).sum()} ({(y == 1).mean():.1%})")

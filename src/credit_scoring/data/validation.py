"""Data validation schemas using Pandera.

This module provides data quality validation for credit scoring inputs,
ensuring data integrity before model training or inference.
"""

from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema


# Define valid categorical values
CHECKING_STATUS_VALUES = ["A11", "A12", "A13", "A14"]
CREDIT_HISTORY_VALUES = ["A30", "A31", "A32", "A33", "A34"]
PURPOSE_VALUES = [f"A4{i}" for i in range(11)]  # A40-A410
SAVINGS_STATUS_VALUES = ["A61", "A62", "A63", "A64", "A65"]
EMPLOYMENT_VALUES = ["A71", "A72", "A73", "A74", "A75"]
PERSONAL_STATUS_VALUES = ["A91", "A92", "A93", "A94", "A95"]
OTHER_PARTIES_VALUES = ["A101", "A102", "A103"]
PROPERTY_VALUES = ["A121", "A122", "A123", "A124"]
OTHER_PLANS_VALUES = ["A141", "A142", "A143"]
HOUSING_VALUES = ["A151", "A152", "A153"]
JOB_VALUES = ["A171", "A172", "A173", "A174"]
TELEPHONE_VALUES = ["A191", "A192"]
FOREIGN_WORKER_VALUES = ["A201", "A202"]


def get_input_schema() -> DataFrameSchema:
    """Get Pandera schema for input validation.

    Returns:
        DataFrameSchema for German Credit dataset format.
    """
    schema = DataFrameSchema(
        columns={
            # Categorical features
            "Attribute1": Column(
                str,
                Check.isin(CHECKING_STATUS_VALUES),
                nullable=False,
                description="Checking account status",
            ),
            "Attribute2": Column(
                int,
                Check.in_range(1, 72),
                nullable=False,
                description="Duration in months",
            ),
            "Attribute3": Column(
                str,
                Check.isin(CREDIT_HISTORY_VALUES),
                nullable=False,
                description="Credit history",
            ),
            "Attribute4": Column(
                str,
                Check.isin(PURPOSE_VALUES),
                nullable=False,
                description="Purpose of loan",
            ),
            "Attribute5": Column(
                float,
                Check.greater_than(0),
                nullable=False,
                description="Credit amount",
            ),
            "Attribute6": Column(
                str,
                Check.isin(SAVINGS_STATUS_VALUES),
                nullable=False,
                description="Savings account status",
            ),
            "Attribute7": Column(
                str,
                Check.isin(EMPLOYMENT_VALUES),
                nullable=False,
                description="Employment duration",
            ),
            "Attribute8": Column(
                int,
                Check.in_range(1, 4),
                nullable=False,
                description="Installment rate",
            ),
            "Attribute9": Column(
                str,
                Check.isin(PERSONAL_STATUS_VALUES),
                nullable=False,
                description="Personal status and sex",
            ),
            "Attribute10": Column(
                str,
                Check.isin(OTHER_PARTIES_VALUES),
                nullable=False,
                description="Other debtors/guarantors",
            ),
            "Attribute11": Column(
                int,
                Check.in_range(1, 4),
                nullable=False,
                description="Present residence since",
            ),
            "Attribute12": Column(
                str,
                Check.isin(PROPERTY_VALUES),
                nullable=False,
                description="Property type",
            ),
            "Attribute13": Column(
                int,
                Check.in_range(18, 100),
                nullable=False,
                description="Age in years",
            ),
            "Attribute14": Column(
                str,
                Check.isin(OTHER_PLANS_VALUES),
                nullable=False,
                description="Other installment plans",
            ),
            "Attribute15": Column(
                str,
                Check.isin(HOUSING_VALUES),
                nullable=False,
                description="Housing type",
            ),
            "Attribute16": Column(
                int,
                Check.in_range(1, 4),
                nullable=False,
                description="Number of existing credits",
            ),
            "Attribute17": Column(
                str,
                Check.isin(JOB_VALUES),
                nullable=False,
                description="Job type",
            ),
            "Attribute18": Column(
                int,
                Check.in_range(1, 2),
                nullable=False,
                description="Number of dependents",
            ),
            "Attribute19": Column(
                str,
                Check.isin(TELEPHONE_VALUES),
                nullable=False,
                description="Telephone",
            ),
            "Attribute20": Column(
                str,
                Check.isin(FOREIGN_WORKER_VALUES),
                nullable=False,
                description="Foreign worker",
            ),
        },
        coerce=True,
        strict=False,
    )

    return schema


def get_target_schema() -> pa.SeriesSchema:
    """Get Pandera schema for target validation.

    Returns:
        SeriesSchema for binary target variable.
    """
    schema = pa.SeriesSchema(
        int,
        checks=[
            Check.isin([0, 1]),
        ],
        nullable=False,
        name="default",
    )

    return schema


def validate_input(
    X: pd.DataFrame,
    raise_exception: bool = True,
) -> Optional[pd.DataFrame]:
    """Validate input data against schema.

    Args:
        X: Input features DataFrame.
        raise_exception: Whether to raise exception on validation failure.

    Returns:
        Validated DataFrame or None if validation fails with raise_exception=False.
    """
    schema = get_input_schema()

    try:
        validated = schema.validate(X, lazy=True)
        print("[OK] Input validation passed!")
        return validated
    except pa.errors.SchemaErrors as e:
        print("[ERROR] Input validation failed!")
        print(f"Errors: {len(e.failure_cases)} issues found")
        print(e.failure_cases)

        if raise_exception:
            raise
        return None


def validate_target(
    y: pd.Series,
    raise_exception: bool = True,
) -> Optional[pd.Series]:
    """Validate target variable against schema.

    Args:
        y: Target Series.
        raise_exception: Whether to raise exception on validation failure.

    Returns:
        Validated Series or None if validation fails with raise_exception=False.
    """
    schema = get_target_schema()

    try:
        validated = schema.validate(y, lazy=True)
        print("[OK] Target validation passed!")
        return validated
    except pa.errors.SchemaErrors as e:
        print("[ERROR] Target validation failed!")
        print(f"Errors: {len(e.failure_cases)} issues found")

        if raise_exception:
            raise
        return None


def generate_validation_report(X: pd.DataFrame, y: pd.Series) -> dict:
    """Generate comprehensive validation report.

    Args:
        X: Input features.
        y: Target variable.

    Returns:
        Dictionary with validation results.
    """
    report = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "missing_values": X.isnull().sum().sum(),
        "duplicate_rows": X.duplicated().sum(),
        "target_distribution": y.value_counts().to_dict(),
    }

    # Validate schemas
    try:
        validate_input(X, raise_exception=True)
        report["input_valid"] = True
    except Exception as e:
        report["input_valid"] = False
        report["input_errors"] = str(e)

    try:
        validate_target(y, raise_exception=True)
        report["target_valid"] = True
    except Exception as e:
        report["target_valid"] = False
        report["target_errors"] = str(e)

    return report

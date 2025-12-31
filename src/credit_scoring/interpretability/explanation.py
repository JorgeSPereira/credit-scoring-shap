"""Customer explanation generator based on SHAP values.

This module generates human-readable explanations for credit decisions,
enabling transparency and compliance with regulations like LGPD.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def generate_customer_explanation(
    shap_values: np.ndarray,
    feature_names: List[str],
    prediction: int,
    probability: float,
    top_n: int = 3,
) -> Dict[str, any]:
    """Generate a human-readable explanation for a credit decision.

    Args:
        shap_values: SHAP values for the prediction (1D array).
        feature_names: List of feature names.
        prediction: Model prediction (0=approved, 1=denied).
        probability: Probability of default.
        top_n: Number of top factors to show.

    Returns:
        Dictionary with explanation components.
    """
    # Create feature importance ranking
    importance = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
        "abs_importance": np.abs(shap_values),
    }).sort_values("abs_importance", ascending=False)

    # Get top factors
    top_factors = importance.head(top_n)

    # Separate positive (increases risk) and negative (decreases risk) factors
    risk_increasing = top_factors[top_factors["shap_value"] > 0]
    risk_decreasing = top_factors[top_factors["shap_value"] < 0]

    # Generate explanation
    decision = "NEGADO" if prediction == 1 else "APROVADO"
    risk_level = _get_risk_level(probability)

    explanation = {
        "decision": decision,
        "probability_default": round(probability * 100, 1),
        "risk_level": risk_level,
        "main_factors": _format_factors(top_factors),
        "risk_factors": _format_factors(risk_increasing),
        "positive_factors": _format_factors(risk_decreasing),
        "summary": _generate_summary(decision, risk_level, top_factors),
    }

    return explanation


def _get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability < 0.3:
        return "BAIXO"
    elif probability < 0.6:
        return "MEDIO"
    else:
        return "ALTO"


def _format_factors(df: pd.DataFrame) -> List[Dict]:
    """Format factors for display."""
    factors = []
    for _, row in df.iterrows():
        # Translate feature names to Portuguese
        feature_pt = _translate_feature(row["feature"])
        impact = "aumenta" if row["shap_value"] > 0 else "diminui"

        factors.append({
            "feature": feature_pt,
            "impact": impact,
            "importance": round(abs(row["shap_value"]), 3),
        })
    return factors


def _translate_feature(feature: str) -> str:
    """Translate feature names to Portuguese."""
    translations = {
        "duration": "Duracao do emprestimo",
        "credit_amount": "Valor do credito",
        "age": "Idade",
        "installment_rate": "Taxa de parcela",
        "present_residence": "Tempo de residencia",
        "num_existing_credits": "Numero de creditos existentes",
        "num_dependents": "Numero de dependentes",
        "checking_status": "Status da conta corrente",
        "credit_history": "Historico de credito",
        "purpose": "Finalidade do emprestimo",
        "savings_status": "Status da poupanca",
        "employment": "Situacao de emprego",
        "personal_status": "Estado civil/genero",
        "other_parties": "Outros fiadores",
        "property_magnitude": "Patrimonio",
        "other_payment_plans": "Outros planos de pagamento",
        "housing": "Tipo de moradia",
        "job": "Tipo de emprego",
        "telephone": "Telefone",
        "foreign_worker": "Trabalhador estrangeiro",
    }

    # Handle encoded features (e.g., "cat__checking_status_A11")
    for key, value in translations.items():
        if key in feature.lower():
            return value

    return feature


def _generate_summary(decision: str, risk_level: str, factors: pd.DataFrame) -> str:
    """Generate a human-readable summary."""
    top_factor = factors.iloc[0]["feature"] if len(factors) > 0 else "diversos fatores"
    top_factor_pt = _translate_feature(top_factor)

    if decision == "NEGADO":
        return (
            f"Credito {decision}. Nivel de risco: {risk_level}. "
            f"O principal fator para esta decisao foi: {top_factor_pt}. "
            f"Para mais informacoes, consulte nosso atendimento."
        )
    else:
        return (
            f"Credito {decision}! Nivel de risco: {risk_level}. "
            f"Seu perfil foi avaliado positivamente. "
            f"Fator mais relevante: {top_factor_pt}."
        )


def print_customer_explanation(explanation: Dict) -> None:
    """Print a formatted customer explanation."""
    print("=" * 60)
    print("DECISAO DE CREDITO")
    print("=" * 60)

    print(f"\nResultado: {explanation['decision']}")
    print(f"Probabilidade de inadimplencia: {explanation['probability_default']}%")
    print(f"Nivel de risco: {explanation['risk_level']}")

    print("\n--- Principais Fatores ---")
    for i, factor in enumerate(explanation["main_factors"], 1):
        impact_symbol = "+" if factor["impact"] == "aumenta" else "-"
        print(f"  {i}. {factor['feature']} ({impact_symbol} risco)")

    print(f"\n--- Resumo ---")
    print(f"  {explanation['summary']}")

    print("\n" + "=" * 60)


def generate_explanation_for_api(
    shap_values: np.ndarray,
    feature_names: List[str],
    prediction: int,
    probability: float,
) -> Dict:
    """Generate explanation suitable for API response.

    Args:
        shap_values: SHAP values for the prediction.
        feature_names: Feature names.
        prediction: Model prediction.
        probability: Probability of default.

    Returns:
        API-ready explanation dictionary.
    """
    explanation = generate_customer_explanation(
        shap_values, feature_names, prediction, probability
    )

    return {
        "decision": explanation["decision"],
        "probability_default_percent": explanation["probability_default"],
        "risk_level": explanation["risk_level"],
        "main_factors": [
            f["feature"] for f in explanation["main_factors"]
        ],
        "summary": explanation["summary"],
    }

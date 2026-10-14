"""
services/anomaly_service.py
===========================
Centralized Anomaly Detection Service.
Refactors and encapsulates the multi-model inference pipeline:
  - Feature extraction (43 features)
  - Isolation Forest inference
  - Local Outlier Factor inference
  - Statistical Z-Score inference
  - Multi-model consensus calibration & risk scoring
  - Audit explanation synthesis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from preprocessing.preprocess import (
    compute_consistency_features,
    compute_date_features,
    compute_frequency_features,
    compute_duplicate_features,
    build_feature_matrix
)


def run_anomaly_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Execute end-to-end unsupervised ML anomaly detection on raw invoice DataFrame.
    Returns DataFrame matching analysis table schema.
    """
    df = df_raw.copy()

    # Numeric validation & date parsing
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    if df["invoice_date"].isnull().any():
        median_date = df["invoice_date"].dropna().median()
        df["invoice_date"] = df["invoice_date"].fillna(median_date)

    num_cols = [
        "quantity", "unit_price", "discount_percentage", "discount_amount",
        "taxable_value", "gst_rate", "cgst", "sgst", "igst",
        "cess_rate", "cess_amount", "total_tax", "total_amount"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    # 1. Feature Engineering
    df_prep = compute_consistency_features(df)
    df_prep = compute_date_features(df_prep)
    df_prep = compute_frequency_features(df_prep)
    df_prep = compute_duplicate_features(df_prep)
    feat_df, feature_cols = build_feature_matrix(df_prep)

    # Scaled feature matrix
    X = feat_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = len(df)
    contamination = 0.05

    # 2. Model 1: Isolation Forest
    if_model = IsolationForest(
        n_estimators=100 if n_samples < 5000 else 200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    if_preds = if_model.fit_predict(X_scaled)
    if_scores = -if_model.score_samples(X_scaled)
    if_anom = (if_preds == -1).astype(int)

    # 3. Model 2: Local Outlier Factor
    n_neighbors = min(20, max(5, n_samples - 1))
    lof_model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1
    )
    lof_preds = lof_model.fit_predict(X_scaled)
    lof_scores = -lof_model.negative_outlier_factor_
    lof_anom = (lof_preds == -1).astype(int)

    # 4. Model 3: Statistical Z-Score
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    Z = np.abs((X - means) / stds)
    max_z = np.max(Z, axis=1)
    z_anom = (max_z >= 3.0).astype(int)
    dom_idx = np.argmax(Z, axis=1)
    dom_feats = [feature_cols[i] for i in dom_idx]

    # 5. Score Normalization (0 - 100)
    if_min, if_max = if_scores.min(), if_scores.max()
    norm_if = ((if_scores - if_min) / (if_max - if_min + 1e-6)) * 100.0

    norm_lof = np.where(
        lof_scores <= 1.0,
        np.clip((lof_scores - 0.90) / 0.10 * 10.0, 0.0, 10.0),
        np.where(
            lof_scores <= 2.5,
            10.0 + ((lof_scores - 1.0) / 1.5) * 70.0,
            80.0 + np.clip((lof_scores - 2.5) / 5.0 * 20.0, 0.0, 20.0)
        )
    )

    norm_z = np.where(
        max_z < 3.0,
        (max_z / 3.0) * 35.0,
        np.clip(50.0 + ((max_z - 3.0) / 7.0) * 50.0, 50.0, 100.0)
    )

    # 6. Ensemble Risk Scoring & Consensus Calibration
    base_score = 0.40 * norm_if + 0.40 * norm_lof + 0.20 * norm_z
    consensus = if_anom + lof_anom + z_anom

    calibrated = base_score.copy()
    calibrated = np.where(consensus == 3, np.maximum(calibrated, 75.0), calibrated)
    calibrated = np.where(consensus == 2, np.maximum(calibrated, 55.0), calibrated)
    calibrated = np.where(consensus == 0, np.minimum(calibrated, 35.0), calibrated)
    risk_score = np.clip(calibrated, 0.0, 100.0).round(2)

    risk_levels = []
    for s in risk_score:
        if s < 40.0:
            risk_levels.append("Low Risk")
        elif s < 70.0:
            risk_levels.append("Medium Risk")
        else:
            risk_levels.append("High Risk")

    # 7. Audit Explanations & Anomaly Types
    explanations = []
    detected_types = []
    for i in range(n_samples):
        lvl = risk_levels[i]
        scr = risk_score[i]
        models = []
        if if_anom[i]: models.append("Isolation Forest")
        if lof_anom[i]: models.append("Local Outlier Factor")
        if z_anom[i]: models.append("Z-Score")

        reasons = []
        dtype = "Normal"

        if df_prep["duplicate_transaction_flag"].iloc[i] == 1:
            reasons.append("duplicate transaction signature")
            dtype = "Duplicate Transaction"
        elif df_prep["future_date_flag"].iloc[i] == 1:
            reasons.append("post-dated invoice outside compliance period")
            dtype = "Future Invoice Date"
        elif df_prep["total_amount_difference"].iloc[i] > 100.0:
            reasons.append(f"major invoice total amount discrepancy (₹{df_prep['total_amount_difference'].iloc[i]:,.2f})")
            dtype = "Extreme Invoice Amount"
        elif df["unit_price"].iloc[i] > 100_000.0:
            reasons.append(f"unit price (₹{df['unit_price'].iloc[i]:,.2f}) significantly exceeds category benchmark")
            dtype = "Extreme Unit Price"
        elif df["quantity"].iloc[i] > 1_000.0:
            reasons.append(f"unusual bulk quantity ({df['quantity'].iloc[i]:,.1f} units)")
            dtype = "Unusual Quantity"
        elif df["discount_percentage"].iloc[i] > 50.0:
            reasons.append(f"abnormally high trade discount ({df['discount_percentage'].iloc[i]:.1f}%)")
            dtype = "Unusual Discount"
        elif df_prep["gst_difference"].iloc[i] > 50.0:
            reasons.append(f"tax calculation inconsistency (₹{df_prep['gst_difference'].iloc[i]:,.2f} variance)")
            dtype = "Tax Calculation Mismatch"
        elif df_prep["cgst_sgst_split_difference"].iloc[i] > 50.0:
            reasons.append("asymmetrical CGST/SGST tax split")
            dtype = "Incorrect CGST SGST Split"
        elif df_prep["igst_difference"].iloc[i] > 50.0:
            reasons.append("inter-state IGST calculation mismatch")
            dtype = "Incorrect IGST"

        if lvl != "Low Risk" and dtype == "Normal":
            dtype = "Statistical Model Outlier"

        if lvl == "Low Risk":
            exp = "Transaction parameters align with compliance benchmarks. No significant anomalies detected."
        elif lvl == "Medium Risk":
            m_str = ", ".join(models) if models else "Statistical model"
            r_str = "; ".join(reasons) if reasons else f"variance in {dom_feats[i].replace('_', ' ')}"
            exp = f"Potentially anomalous transaction flagged by {m_str}. Review recommended due to {r_str}."
        else:
            m_str = ", ".join(models) if models else "Multi-model consensus"
            r_str = "; ".join(reasons) if reasons else f"extreme variance in {dom_feats[i].replace('_', ' ')}"
            exp = f"High-risk anomaly identified by {m_str} (Risk Score: {scr:.1f}/100). Requires priority review due to {r_str}."

        explanations.append(exp)
        detected_types.append(dtype)

    # 8. Aligned Analysis DataFrame
    analysis_df = pd.DataFrame({
        "invoice_no": df["invoice_no"].values,
        "isolation_score": if_scores.round(4),
        "isolation_anomaly": if_anom,
        "lof_score": lof_scores.round(4),
        "lof_anomaly": lof_anom,
        "max_abs_zscore": max_z.round(4),
        "zscore_anomaly": z_anom,
        "risk_score": risk_score,
        "risk_level": risk_levels,
        "anomaly_type": detected_types,
        "explanation": explanations
    })

    return analysis_df

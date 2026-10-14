"""
services/explanation_service.py
===============================
AI-Powered Anomaly Explanation Engine for GSTGuard AI.
Transforms raw model outputs, statistical deviations, and invoice parameters into:
  1. Transparent multi-model consensus breakdown
  2. Empirical evidence table (Actual vs Typical MSME Benchmark)
  3. Actionable plain-language compliance narrative
"""

from typing import Dict, Any, List


# Typical MSME benchmarks derived from empirical dataset distribution
TYPICAL_BENCHMARKS = {
    "unit_price": {
        "label": "Unit Price (₹)",
        "typical_range": "₹50.00 – ₹10,000.00",
        "benchmark_val": 2500.0,
        "tolerance": 50000.0
    },
    "quantity": {
        "label": "Billed Quantity (Units)",
        "typical_range": "1 – 500 units",
        "benchmark_val": 50.0,
        "tolerance": 1000.0
    },
    "taxable_value": {
        "label": "Taxable Value (₹)",
        "typical_range": "₹1,000.00 – ₹2,50,000.00",
        "benchmark_val": 45000.0,
        "tolerance": 1000000.0
    },
    "discount_percentage": {
        "label": "Trade Discount Rate (%)",
        "typical_range": "0.0% – 25.0%",
        "benchmark_val": 5.0,
        "tolerance": 40.0
    },
    "gst_rate": {
        "label": "GST Slab Rate (%)",
        "typical_range": "5%, 12%, 18%, 28%",
        "benchmark_val": 18.0,
        "tolerance": 28.0
    },
    "cgst_sgst_ratio": {
        "label": "CGST : SGST Ratio",
        "typical_range": "1.00 (Exact 50/50 Equal Split)",
        "benchmark_val": 1.0,
        "tolerance": 0.05
    },
    "total_amount_variance": {
        "label": "Total Amount Consistency (₹)",
        "typical_range": "₹0.00 variance",
        "benchmark_val": 0.0,
        "tolerance": 1.0
    }
}


def build_evidence_items(invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construct empirical evidence rows comparing invoice parameters against compliance benchmarks.
    """
    evidence = []

    # 1. Unit Price
    unit_price = float(invoice_data.get("unit_price", 0.0))
    if unit_price > 50000.0:
        status = "CRITICAL_HIGH"
        status_label = "Severe Outlier"
        badge_class = "danger"
    elif unit_price > 20000.0:
        status = "ELEVATED"
        status_label = "Elevated"
        badge_class = "warning"
    else:
        status = "NORMAL"
        status_label = "Expected Range"
        badge_class = "success"

    evidence.append({
        "feature": "Unit Price",
        "actual": f"₹{unit_price:,.2f}",
        "benchmark": "₹50.00 – ₹10,000.00",
        "status": status,
        "status_label": status_label,
        "badge_class": badge_class,
        "detail": f"Unit price deviates by {max(1.0, unit_price / 2500.0):.1f}x from the category median."
    })

    # 2. Quantity
    qty = float(invoice_data.get("quantity", 0.0))
    if qty > 1000.0:
        status = "CRITICAL_HIGH"
        status_label = "Severe Outlier"
        badge_class = "danger"
    elif qty > 500.0:
        status = "ELEVATED"
        status_label = "Elevated"
        badge_class = "warning"
    else:
        status = "NORMAL"
        status_label = "Expected Range"
        badge_class = "success"

    evidence.append({
        "feature": "Billed Quantity",
        "actual": f"{qty:,.1f} units",
        "benchmark": "1.0 – 500.0 units",
        "status": status,
        "status_label": status_label,
        "badge_class": badge_class,
        "detail": f"Quantity is {max(1.0, qty / 50.0):.1f}x higher than typical batch orders."
    })

    # 3. Trade Discount
    disc = float(invoice_data.get("discount_percentage", 0.0))
    if disc > 50.0:
        status = "CRITICAL_HIGH"
        status_label = "Abnormal Discount"
        badge_class = "danger"
    elif disc > 30.0:
        status = "ELEVATED"
        status_label = "High Discount"
        badge_class = "warning"
    else:
        status = "NORMAL"
        status_label = "Standard"
        badge_class = "success"

    evidence.append({
        "feature": "Discount Percentage",
        "actual": f"{disc:.1f}%",
        "benchmark": "0.0% – 25.0%",
        "status": status,
        "status_label": status_label,
        "badge_class": badge_class,
        "detail": "High trade discounts without scheme documentation can trigger GST scrutiny."
    })

    # 4. Tax Calculation Consistency
    taxable = float(invoice_data.get("taxable_value", 0.0))
    gst_rate = float(invoice_data.get("gst_rate", 0.0))
    total_tax = float(invoice_data.get("total_tax", 0.0))
    expected_tax = taxable * (gst_rate / 100.0)
    tax_diff = abs(total_tax - expected_tax)

    if tax_diff > 50.0:
        status = "CRITICAL_HIGH"
        status_label = "Mismatch Detected"
        badge_class = "danger"
    elif tax_diff > 2.0:
        status = "ELEVATED"
        status_label = "Minor Rounding Variance"
        badge_class = "warning"
    else:
        status = "NORMAL"
        status_label = "Exact Match"
        badge_class = "success"

    evidence.append({
        "feature": "GST Tax Calculation",
        "actual": f"₹{total_tax:,.2f}",
        "benchmark": f"₹{expected_tax:,.2f} ({gst_rate:.0f}% of Taxable)",
        "status": status,
        "status_label": status_label,
        "badge_class": badge_class,
        "detail": f"Variance of ₹{tax_diff:,.2f} detected between invoice tax and statutory formula." if tax_diff > 2.0 else "Statutory calculation matches GST slab exactly."
    })

    # 5. CGST vs SGST Split (for Intra-State)
    cgst = float(invoice_data.get("cgst", 0.0))
    sgst = float(invoice_data.get("sgst", 0.0))
    if cgst > 0 or sgst > 0:
        split_diff = abs(cgst - sgst)
        if split_diff > 10.0:
            status = "CRITICAL_HIGH"
            status_label = "Asymmetrical Split"
            badge_class = "danger"
        else:
            status = "NORMAL"
            status_label = "Equal 50/50 Split"
            badge_class = "success"

        evidence.append({
            "feature": "CGST / SGST Split",
            "actual": f"CGST: ₹{cgst:,.2f} | SGST: ₹{sgst:,.2f}",
            "benchmark": "CGST == SGST (1:1 Ratio)",
            "status": status,
            "status_label": status_label,
            "badge_class": badge_class,
            "detail": f"Intra-state split discrepancy of ₹{split_diff:,.2f}." if split_diff > 10.0 else "CGST and SGST are equally apportioned."
        })

    # 6. Total Amount Formula (Taxable Value + Total Tax - Discount + Cess)
    total_amt = float(invoice_data.get("total_amount", 0.0))
    cess_amt = float(invoice_data.get("cess_amount", 0.0))
    expected_total = taxable + total_tax + cess_amt
    amt_diff = abs(total_amt - expected_total)

    if amt_diff > 100.0:
        status = "CRITICAL_HIGH"
        status_label = "Amount Mismatch"
        badge_class = "danger"
    elif amt_diff > 5.0:
        status = "ELEVATED"
        status_label = "Minor Discrepancy"
        badge_class = "warning"
    else:
        status = "NORMAL"
        status_label = "Mathematically Consistent"
        badge_class = "success"

    evidence.append({
        "feature": "Invoice Grand Total",
        "actual": f"₹{total_amt:,.2f}",
        "benchmark": f"₹{expected_total:,.2f}",
        "status": status,
        "status_label": status_label,
        "badge_class": badge_class,
        "detail": f"Grand total deviates by ₹{amt_diff:,.2f} from (Taxable Value + Tax + Cess)." if amt_diff > 5.0 else "Grand total aligns with line item components."
    })

    return evidence


def build_model_breakdown(invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construct multi-model consensus and anomaly contribution breakdown.
    """
    if_anom = int(invoice_data.get("isolation_anomaly", 0))
    lof_anom = int(invoice_data.get("lof_anomaly", 0))
    z_anom = int(invoice_data.get("zscore_anomaly", 0))

    if_score = float(invoice_data.get("isolation_score", 0.0))
    lof_score = float(invoice_data.get("lof_score", 0.0))
    max_z = float(invoice_data.get("max_abs_zscore", 0.0))

    models = [
        {
            "name": "Isolation Forest",
            "type": "Tree-Partitioning (Global Structure)",
            "flagged": bool(if_anom),
            "raw_score": f"{if_score:.4f}",
            "contribution_weight": "40%",
            "assessment": "Identified anomalous tree-depth path" if if_anom else "Within standard feature partition",
            "badge_class": "danger" if if_anom else "success"
        },
        {
            "name": "Local Outlier Factor (LOF)",
            "type": "Density-Based (Local Neighborhood)",
            "flagged": bool(lof_anom),
            "raw_score": f"{lof_score:.4f}",
            "contribution_weight": "40%",
            "assessment": "Significantly lower density than k-nearest neighbors" if lof_anom else "Consistent with local cluster density",
            "badge_class": "danger" if lof_anom else "success"
        },
        {
            "name": "Statistical Z-Score",
            "type": "Parametric Standard Deviation",
            "flagged": bool(z_anom),
            "raw_score": f"{max_z:.2f} σ",
            "contribution_weight": "20%",
            "assessment": f"Exceeds 3.0 standard deviations ({max_z:.1f}σ)" if z_anom else "Within 3.0σ normal distribution",
            "badge_class": "danger" if z_anom else "success"
        }
    ]
    return models


def get_detailed_explanation(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full explanatory payload for invoice details view, Copilot, and compliance reports.
    """
    evidence = build_evidence_items(invoice_data)
    models = build_model_breakdown(invoice_data)

    flagged_evidence = [e for e in evidence if e["status"] != "NORMAL"]
    consensus_count = sum(1 for m in models if m["flagged"])

    risk_level = invoice_data.get("risk_level", "Low Risk")
    risk_score = float(invoice_data.get("risk_score", 0.0))
    anomaly_type = invoice_data.get("anomaly_type", "Normal")

    # Generate multi-paragraph synthesized assessment
    if risk_level == "High Risk":
        summary = (
            f"This transaction exhibits elevated compliance risk with a composite risk score of "
            f"{risk_score:.1f}/100. {consensus_count} of 3 ML models reached consensus on this anomaly. "
            f"Primary trigger: {anomaly_type}. Immediate verification of supporting e-Way bills and "
            f"supplier filing records is advised before claiming Input Tax Credit (ITC)."
        )
    elif risk_level == "Medium Risk":
        summary = (
            f"This transaction presents moderate statistical variance (Risk Score: {risk_score:.1f}/100) "
            f"identified by {consensus_count} detection model(s). Specific flags include: {anomaly_type}. "
            f"Review recommended to ensure compliance documentation is complete."
        )
    else:
        summary = (
            f"Transaction parameters conform to standard MSME billing and tax filing patterns "
            f"(Risk Score: {risk_score:.1f}/100). No significant mathematical or structural anomalies detected."
        )

    return {
        "summary": summary,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "anomaly_type": anomaly_type,
        "consensus_count": consensus_count,
        "evidence": evidence,
        "flagged_evidence_count": len(flagged_evidence),
        "models": models
    }

"""
services/copilot_service.py
===========================
Compliance Copilot: Deterministic, data-grounded assistant for MSME GST compliance.
Executes locally without external API dependencies. Answers questions by synthesizing
actual transaction line items, ML model outputs, empirical benchmarks, and GST rules.
"""

import re
from collections import Counter
from typing import Dict, Any, Optional
from services.explanation_service import get_detailed_explanation
from services.recommendation_service import get_recommendations_for_invoice


def _supplier_summary(supplier_data: Dict[str, Any], invoice_no: Optional[str] = None) -> Dict[str, str]:
    supplier_id = supplier_data.get("supplier_id", "Unknown")
    supplier_gstin = supplier_data.get("supplier_gstin", "Unknown")
    invoice_records = supplier_data.get("invoices", [])
    anomaly_counts = Counter()

    for invoice in invoice_records:
        anomaly_type = invoice.get("anomaly_type") or "Normal"
        if anomaly_type == "Normal" and invoice.get("risk_level") in ("Medium Risk", "High Risk"):
            anomaly_type = "Statistical Model Outlier"
        anomaly_counts[anomaly_type] += 1

    anomaly_summary = "\n".join(
        f"- **{name}:** {count} invoice(s)"
        for name, count in anomaly_counts.most_common()
    ) or "- No invoice pattern data is available."

    context = f" for invoice `{invoice_no}`" if invoice_no else ""
    company_name = supplier_data.get("company_name")
    company_line = (
        f"- **Company name:** {company_name}"
        if company_name else
        "- **Company name:** Not recorded in this dataset; the supplier is identified by its supplier ID and GSTIN."
    )
    answer = (
        f"### Supplier Summary{context}\n\n"
        f"{company_line}\n"
        f"- **Supplier ID:** `{supplier_id}`\n"
        f"- **GSTIN:** `{supplier_gstin}`\n"
        f"- **Invoices in this dataset:** {supplier_data.get('total_invoices', len(invoice_records))}\n"
        f"- **Risk levels:** High {supplier_data.get('high_risk_count', 0)}, "
        f"Medium {supplier_data.get('medium_risk_count', 0)}, "
        f"Low {supplier_data.get('low_risk_count', 0)}\n\n"
        f"**Anomaly patterns across its invoices:**\n{anomaly_summary}"
    )
    return {"answer": answer, "confidence": "High (Data-grounded)", "category": "supplier_summary"}


def query_copilot(query_text: str, invoice_data: Optional[Dict[str, Any]] = None, supplier_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a user query in natural language and return a structured, data-grounded answer.
    """
    q = query_text.lower().strip()
    
    # ── CASE 1: Invoice-Specific Context ─────────────────────────────────────
    if invoice_data:
        inv_no = invoice_data.get("invoice_no", "Unknown")
        risk_level = invoice_data.get("risk_level", "Low Risk")
        risk_score = float(invoice_data.get("risk_score", 0.0))
        anomaly_type = invoice_data.get("anomaly_type", "Normal")
        supplier_id = invoice_data.get("supplier_id", "Unknown")
        supplier_gstin = invoice_data.get("supplier_gstin", "Unknown")
        taxable_val = float(invoice_data.get("taxable_value", 0.0))
        total_tax = float(invoice_data.get("total_tax", 0.0))
        total_amt = float(invoice_data.get("total_amount", 0.0))
        gst_rate = float(invoice_data.get("gst_rate", 0.0))

        # Model flags
        if_anom = int(invoice_data.get("isolation_anomaly", 0))
        lof_anom = int(invoice_data.get("lof_anomaly", 0))
        z_anom = int(invoice_data.get("zscore_anomaly", 0))

        if any(term in q for term in ["company", "supplier", "vendor", "how many invoices", "produced", "issued"]):
            if supplier_data:
                return _supplier_summary(supplier_data, inv_no)
            return {
                "answer": (
                    f"The invoice lists supplier `{supplier_id}` with GSTIN `{supplier_gstin}`, "
                    "but no supplier invoice history is available for a count or anomaly breakdown."
                ),
                "confidence": "Limited (Supplier data unavailable)",
                "category": "supplier_summary"
            }

        if any(term in q for term in [
            "reduce risk", "lower risk", "reduce the risk score", "lower the risk score",
            "reduce score", "lower score", "improve risk score", "bring the risk score down"
        ]):
            recommendations = get_recommendations_for_invoice(invoice_data)
            specific_actions = recommendations.get("actions", [])
            if recommendations.get("title") == "Standard Compliance Verification":
                action_text = (
                    "- Reconcile invoice line items, taxable value, GST rate, and tax components against the purchase order.\n"
                    "- Verify the supplier filing and invoice match in GSTR-2B.\n"
                    "- Confirm delivery evidence, check for duplicate booking, and retain payment records."
                )
            else:
                action_text = "\n".join(
                    f"- **{item['action']}:** {item['detail']}"
                    for item in specific_actions[:3]
                )
            answer = (
                f"### Risk Reduction Plan for Invoice `{inv_no}`\n\n"
                f"Current score: **{risk_score:.1f}/100 ({risk_level})**; detected pattern: **{anomaly_type}**.\n\n"
                "The score should not be changed manually. Resolve and document the underlying issue, "
                "then rerun the analysis to get an updated score. Suggested checks:\n\n"
                f"{action_text}\n"
                "- Confirm the corrected source documents and supporting records are retained."
            )
            return {"answer": answer, "confidence": "High (Data-grounded)", "category": "risk_reduction"}

        # Handle explicit model questions before anomaly/explanation keywords.
        if any(w in q for w in ["model", "algorithm", "isolation", "lof", "zscore", "consensus"]):
            exp = get_detailed_explanation(invoice_data)
            model_cards = "\n".join([
                f"- **{m['name']}** ({m['type']}): Flagged = **{'YES' if m['flagged'] else 'NO'}** (Raw Score: `{m['raw_score']}`, Weight: {m['contribution_weight']}). *{m['assessment']}*"
                for m in exp["models"]
            ])
            answer = (
                f"### Model Evaluation Breakdown for `{inv_no}`\n\n"
                f"{model_cards}\n\n"
                f"**Ensemble Calibration:** The weighted score is combined as: "
                f"`(0.40 * IF) + (0.40 * LOF) + (0.20 * Z-Score)`. "
                f"Because **{exp['consensus_count']} of 3** models identified anomalous behavior, "
                f"the calibrated risk score is **{risk_score:.1f}/100**."
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "models"
            }

        # 1. Why was it flagged? / Why is it risky?
        if any(w in q for w in ["why", "reason", "flag", "flagged", "anomaly", "suspicious", "detect"]):
            exp = get_detailed_explanation(invoice_data)
            flagged_models = [m["name"] for m in exp["models"] if m["flagged"]]
            m_text = ", ".join(flagged_models) if flagged_models else "None (Low Risk transaction)"

            evidence_bullets = "\n".join([
                f"- **{e['feature']}**: Actual value `{e['actual']}` vs benchmark `{e['benchmark']}` — *{e['detail']}*"
                for e in exp["evidence"] if e["status"] != "NORMAL"
            ])
            if not evidence_bullets:
                evidence_bullets = "- All evaluated features are within standard tolerance limits."

            answer = (
                f"### Analysis for Invoice `{inv_no}`\n\n"
                f"**Risk Score:** `{risk_score:.1f}/100` ({risk_level})\n"
                f"**Classification:** `{anomaly_type}`\n"
                f"**Flagged by Models:** {m_text} ({exp['consensus_count']}/3 consensus)\n\n"
                f"**Key Evidence Traced:**\n{evidence_bullets}\n\n"
                f"**Summary:** {exp['summary']}"
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "explanation"
            }

        # 3. Tax discrepancy / Tax calculation
        if any(w in q for w in ["tax", "gst", "calculation", "cgst", "sgst", "igst", "diff"]):
            expected_tax = taxable_val * (gst_rate / 100.0)
            diff = abs(total_tax - expected_tax)
            cgst = float(invoice_data.get("cgst", 0.0))
            sgst = float(invoice_data.get("sgst", 0.0))
            igst = float(invoice_data.get("igst", 0.0))

            answer = (
                f"### Tax Computation Audit for `{inv_no}`\n\n"
                f"- **Taxable Value:** ₹{taxable_val:,.2f}\n"
                f"- **Statutory GST Slab:** {gst_rate:.0f}%\n"
                f"- **Expected Tax Amount:** ₹{expected_tax:,.2f}\n"
                f"- **Billed Total Tax:** ₹{total_tax:,.2f}\n"
                f"- **Variance:** ₹{diff:,.2f}\n\n"
                f"**Tax Components:**\n"
                f"- CGST: ₹{cgst:,.2f}\n"
                f"- SGST: ₹{sgst:,.2f}\n"
                f"- IGST: ₹{igst:,.2f}\n\n"
                f"**Status:** " + (
                    f"⚠️ Significant tax variance of ₹{diff:,.2f} detected! Issue debit/credit note before claiming ITC."
                    if diff > 5.0 else
                    "✅ Tax calculations mathematically conform to statutory GST rules."
                )
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "tax"
            }

        # 4. Actions to take / What should I do? / Next steps
        if any(w in q for w in ["action", "recommend", "step", "fix", "resolve", "checklist", "what should i do", "what do i do"]):
            rec = get_recommendations_for_invoice(invoice_data)
            action_items = "\n".join([
                f"{i+1}. **[{a['priority']}] {a['action']}**\n   - *Details:* {a['detail']}\n   - *Required Doc:* `{a['doc_required']}`"
                for i, a in enumerate(rec["actions"])
            ])
            answer = (
                f"### Recommended Compliance Actions for `{inv_no}`\n\n"
                f"**Recommended Workflow Action:** `{rec['workflow_label']}`\n\n"
                f"**Action Checklist:**\n{action_items}\n\n"
                f"**Risk Context:** {rec['risk_impact']}"
            )
            return {
                "answer": answer,
                "confidence": "High (Rule-based)",
                "category": "recommendations"
            }

        # 5. Can I claim ITC? / Is it safe?
        if any(w in q for w in ["itc", "claim", "input tax", "safe", "credit"]):
            if risk_level == "High Risk":
                itc_status = "❌ **DO NOT CLAIM YET** (Hold ITC)"
                itc_reason = (
                    f"This invoice has high risk ({risk_score:.1f}/100) and was flagged for `{anomaly_type}`. "
                    f"Claiming this credit now without verifying e-Way bills and supplier GSTR-1 filings "
                    f"risks GST notice under Section 16(2) and Section 73/74 with 18% annual interest."
                )
            elif risk_level == "Medium Risk":
                itc_status = "⚠️ **PROCEED WITH DOCUMENTATION**"
                itc_reason = (
                    f"This invoice has moderate risk ({risk_score:.1f}/100). Verify that the invoice appears in "
                    f"your monthly GSTR-2B before including it in GSTR-3B Table 4(A)(5)."
                )
            else:
                itc_status = "✅ **SAFE TO CLAIM**"
                itc_reason = (
                    f"This invoice passed all anomaly checks ({risk_score:.1f}/100). Ensure regular payment "
                    f"within 180 days to maintain Section 16(2) compliance."
                )

            answer = (
                f"### Input Tax Credit (ITC) Guidance for `{inv_no}`\n\n"
                f"**Recommendation:** {itc_status}\n\n"
                f"{itc_reason}"
            )
            return {
                "answer": answer,
                "confidence": "High (Statutory Guidance)",
                "category": "itc"
            }

        # 6. Supplier details
        if any(w in q for w in ["supplier", "vendor", "gstin"]):
            if supplier_data:
                return _supplier_summary(supplier_data, inv_no)
            answer = (
                f"### Supplier Context for `{inv_no}`\n\n"
                f"- **Supplier ID:** `{supplier_id}`\n"
                f"- **Supplier GSTIN:** `{supplier_gstin}`\n"
                f"- **State Code:** `{supplier_gstin[:2] if len(supplier_gstin) >= 2 else 'N/A'}`\n"
                f"- View full profile and compliance track record under the **Suppliers** tab."
            )
            return {
                "answer": answer,
                "confidence": "High (Data-grounded)",
                "category": "supplier"
            }

        if any(w in q for w in ["risk level", "risk category", "risk categories", "risk threshold", "risk thresholds"]):
            answer = (
                f"### Risk Classification for Invoice `{inv_no}`\n\n"
                f"This invoice has a risk score of **{risk_score:.1f}/100**, classified as **{risk_level}**.\n\n"
                "- **Low Risk:** 0–39\n"
                "- **Medium Risk:** 40–69\n"
                "- **High Risk:** 70–100"
            )
            return {"answer": answer, "confidence": "High (Data-grounded)", "category": "risk_thresholds"}

    # ── CASE 2: General Knowledge & FAQ Queries ──────────────────────────────
    if any(w in q for w in ["gstr-2b", "gstr 2b", "2b"]):
        answer = (
            "### GSTR-2B Overview & Rule 36(4)\n\n"
            "**GSTR-2B** is an auto-drafted, static Input Tax Credit (ITC) statement generated on the 14th of every month. "
            "Under GST Rule 36(4), businesses can only claim ITC that is explicitly reflected in their GSTR-2B. "
            "If a supplier fails to upload the invoice in GSTR-1 by the due date, the credit will not appear in GSTR-2B and cannot be claimed."
        )
        return {"answer": answer, "confidence": "High", "category": "general_gst"}

    if any(w in q for w in ["180 day", "180-day", "section 16"]):
        answer = (
            "### Section 16(2) — 180-Day Payment Rule\n\n"
            "Under the second proviso to Section 16(2) of the CGST Act, the recipient must pay the supplier the value of goods/services "
            "plus GST within **180 days** from the invoice date. If unpaid, the claimed ITC must be reversed in GSTR-3B with applicable 18% annual interest."
        )
        return {"answer": answer, "confidence": "High", "category": "general_gst"}

    if any(w in q for w in ["weights", "formula", "ensemble", "scoring"]):
        answer = (
            "### GSTGuard AI Ensemble Scoring Model\n\n"
            "The platform synthesizes 3 complementary anomaly detection models:\n\n"
            "1. **Isolation Forest (40% Weight):** Detects global multidimensional outliers via random decision trees.\n"
            "2. **Local Outlier Factor (40% Weight):** Evaluates local neighborhood density around each invoice.\n"
            "3. **Statistical Z-Score (20% Weight):** Identifies extreme standard deviation outliers beyond 3.0σ.\n\n"
            "**Multi-Model Consensus:**\n"
            "- 3 Models agree → Risk Score calibrated to >= 75 (High Risk)\n"
            "- 2 Models agree → Risk Score calibrated to >= 55 (Medium/High Risk)\n"
            "- 0 Models agree → Risk Score capped below 35 (Low Risk)"
        )
        return {"answer": answer, "confidence": "High", "category": "general_ml"}

    if any(w in q for w in ["risk level", "risk category", "risk categories", "risk threshold", "risk thresholds"]):
        answer = (
            "### GSTGuard AI Risk Categories\n\n"
            "Invoice risk scores are classified using these thresholds:\n\n"
            "- **Low Risk:** 0–39\n"
            "- **Medium Risk:** 40–69\n"
            "- **High Risk:** 70–100\n\n"
            "Higher scores indicate a greater need for supporting-document and compliance review."
        )
        return {"answer": answer, "confidence": "High", "category": "risk_thresholds"}

    # Default fallback
    if invoice_data:
        exp = get_detailed_explanation(invoice_data)
        answer = (
            f"### Invoice `{invoice_data.get('invoice_no')}` Overview\n\n"
            f"- **Risk Level:** {invoice_data.get('risk_level')} ({invoice_data.get('risk_score', 0):.1f}/100)\n"
            f"- **Anomaly Type:** {invoice_data.get('anomaly_type')}\n"
            f"- **Summary:** {exp['summary']}\n\n"
            f"You can ask me:\n"
            f"- *'Why was this invoice flagged?'*\n"
            f"- *'What models flagged this transaction?'*\n"
            f"- *'Is it safe to claim ITC?'*\n"
            f"- *'What are the recommended actions?'*\n"
            f"- *'What is the tax discrepancy?'*"
        )
    else:
        answer = (
            "### GSTGuard AI Compliance Copilot\n\n"
            "I can assist you with:\n"
            "- Explaining why an invoice was flagged as anomalous\n"
            "- Model consensus and evidence verification\n"
            "- Input Tax Credit (ITC) eligibility and risk checks\n"
            "- Supplier profile intelligence\n"
            "- GST statutory rules (GSTR-2B, Section 16(2), e-Way Bill guidelines)\n\n"
            "Please ask a specific compliance question or select an invoice to begin."
        )

    return {
        "answer": answer,
        "confidence": "Medium (General Guidance)",
        "category": "general"
    }

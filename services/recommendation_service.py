"""
services/recommendation_service.py
==================================
MSME Compliance Action Center recommendations.
Generates context-aware, practical compliance action checklists based on
the detected anomaly type, model consensus, and statutory GST regulations.
"""

from typing import Dict, Any, List


# Recommendation patterns mapped by anomaly type
ANOMALY_ACTION_MAP = {
    "Extreme Unit Price": {
        "title": "Unit Price Outlier Verification",
        "description": "The billed unit price significantly exceeds typical category benchmarks, presenting risk of invoice over-valuation or misclassification.",
        "risk_impact": "High risk of GST department scrutiny under Section 15 Valuation Rules; potential ITC disallowance if goods value is inflated.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Verify Original Purchase Order & Rate Contract",
                "detail": "Cross-check billed rates against the approved vendor purchase agreement or master price list.",
                "doc_required": "Purchase Order (PO), Rate Contract"
            },
            {
                "priority": "HIGH",
                "action": "Inspect Physical Delivery Challan & Goods Receipt Note (GRN)",
                "detail": "Confirm that high-value specialized items were physically received, inspected, and logged in warehouse records.",
                "doc_required": "Signed Goods Receipt Note (GRN), Inspection Sheet"
            },
            {
                "priority": "MEDIUM",
                "action": "Check HSN Code Classification",
                "detail": "Confirm that the item isn't misclassified under an incorrect 6/8-digit HSN code with an improper valuation threshold.",
                "doc_required": "GST HSN Schedule, Technical Specification Sheet"
            }
        ]
    },
    "Extreme Invoice Amount": {
        "title": "Substantial Invoice Total Reconciliation",
        "description": "Invoice grand total diverges significantly from normal transaction sizes or has mathematical summation discrepancies.",
        "risk_impact": "Major discrepancy could result in automated mismatch alerts between GSTR-1 and GSTR-3B, risking provisional ITC blocking.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Perform Line-Item Summation Check",
                "detail": "Verify that (Taxable Value + CGST + SGST + IGST + Cess) exactly equals Grand Total without unexplained surcharges.",
                "doc_required": "Detailed Itemized Invoice"
            },
            {
                "priority": "HIGH",
                "action": "Verify e-Way Bill Part A and Part B",
                "detail": "Mandatory for consignments over ₹50,000. Verify the e-Way bill number, validity date, and vehicle tracking data.",
                "doc_required": "Valid e-Way Bill (with matching Invoice No)"
            },
            {
                "priority": "MEDIUM",
                "action": "Validate Bank Payment Trails",
                "detail": "Ensure payment is made via approved banking channels (NEFT/RTGS) to meet Section 16(2) 180-day payment compliance.",
                "doc_required": "Bank Statement / Payment Advice"
            }
        ]
    },
    "Tax Calculation Mismatch": {
        "title": "Statutory Tax Formula Reconciliation",
        "description": "Computed tax amounts deviate from the statutory GST rate applied to the taxable turnover.",
        "risk_impact": "Direct violation of GST Rule 46. Claiming improper tax amounts will generate GSTR-2B mismatch notices.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Hold ITC Claim Until Amendment",
                "detail": "Do not claim Input Tax Credit on this invoice until the supplier issues a corrective Credit/Debit note.",
                "doc_required": "Reconciliation Memo"
            },
            {
                "priority": "HIGH",
                "action": "Issue Notice to Supplier for GSTR-1 Correction",
                "detail": "Notify the vendor to rectify the tax liability figures in their next GSTR-1 return filing.",
                "doc_required": "Vendor Communication Letter / Email"
            },
            {
                "priority": "MEDIUM",
                "action": "Obtain Revised / Rectified Tax Invoice",
                "detail": "Request an amended invoice reflecting the accurate mathematical tax computation.",
                "doc_required": "Corrected Invoice / Credit Note"
            }
        ]
    },
    "Incorrect CGST SGST Split": {
        "title": "Intra-State CGST / SGST Balancing",
        "description": "CGST and SGST amounts are unequal for an intra-state supply, which requires an exact 50/50 statutory split.",
        "risk_impact": "Automated GST portal validation rejects asymmetrical CGST/SGST filings during GSTR-3B auto-population.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Verify Place of Supply vs Supplier Registration",
                "detail": "Confirm that both supplier and recipient states match and that the transaction is indeed intra-state.",
                "doc_required": "GSTIN State Code Verification (First 2 digits)"
            },
            {
                "priority": "HIGH",
                "action": "Request Immediate Split Correction",
                "detail": "Instruct supplier accounts to reissue invoice with equal division of the total GST rate.",
                "doc_required": "Corrective Debit/Credit Note"
            }
        ]
    },
    "Incorrect IGST": {
        "title": "Inter-State IGST Apportionment Verification",
        "description": "Inter-state IGST calculation mismatch or unexpected IGST levy on what appears to be local supply.",
        "risk_impact": "Wrong tax head credit claimed (IGST instead of CGST/SGST) may incur interest liabilities under Section 50.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Validate Place of Supply (Section 10/12 of IGST Act)",
                "detail": "Ascertain whether the goods or services were delivered across state boundaries.",
                "doc_required": "Delivery Address / Lorry Receipt (LR)"
            },
            {
                "priority": "HIGH",
                "action": "Verify GSTR-2B Tax Head Mapping",
                "detail": "Ensure that the invoice appears under Table 4(A)(5) for IGST in GSTR-2B.",
                "doc_required": "GSTR-2B Portal Download"
            }
        ]
    },
    "Duplicate Transaction": {
        "title": "Duplicate Invoice Detection & ITC Guard",
        "description": "Identical supplier, buyer, line-items, and amount detected on multiple invoice numbers.",
        "risk_impact": "Severe compliance penalty for claiming double Input Tax Credit on the same underlying supply.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Block Duplicate ITC Entry in ERP",
                "detail": "Ensure your accounting/ERP system has marked this transaction as a potential duplicate to prevent double payment.",
                "doc_required": "ERP Journal Voucher"
            },
            {
                "priority": "HIGH",
                "action": "Cross-Verify GSTR-2B Auto-Populated List",
                "detail": "Check whether the supplier erroneously reported the invoice twice in their GSTR-1.",
                "doc_required": "GSTR-2B Supplier Summary"
            },
            {
                "priority": "MEDIUM",
                "action": "Obtain Cancellation Confirmation from Vendor",
                "detail": "If the duplicate was issued by mistake, obtain formal written cancellation.",
                "doc_required": "Vendor Cancellation Letter"
            }
        ]
    },
    "Future Invoice Date": {
        "title": "Post-Dated Invoice Compliance Review",
        "description": "The invoice date is recorded beyond the current authorized tax compliance cutoff period.",
        "risk_impact": "Invoices dated in future tax periods cannot be claimed in current GSTR-3B filings under Section 16(2).",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Defer ITC to Valid Tax Period",
                "detail": "Move the invoice into the appropriate future tax month ledger.",
                "doc_required": "Monthly ITC Register"
            },
            {
                "priority": "MEDIUM",
                "action": "Confirm Actual Date of Physical Dispatch",
                "detail": "Check whether the invoice date was a clerical typo versus actual future supply date.",
                "doc_required": "Transporter Consignment Note"
            }
        ]
    },
    "Unusual Quantity": {
        "title": "Bulk Volume & Logistics Audit",
        "description": "Consignment quantity significantly exceeds typical volume thresholds for the product category.",
        "risk_impact": "Risks scrutiny for fictitious billing or circular trading if transport infrastructure cannot substantiate the volume.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Inspect Weighbridge & Toll Receipts",
                "detail": "Verify that commercial vehicles had appropriate payload capacity for the recorded quantity.",
                "doc_required": "Weighbridge Slips, Fastag / Toll Receipts"
            },
            {
                "priority": "MEDIUM",
                "action": "Cross-Check Warehouse Inward Capacity",
                "detail": "Confirm warehouse space was available and that goods were logged in inventory records.",
                "doc_required": "Stock Ledger / Bin Card"
            }
        ]
    },
    "Unusual Discount": {
        "title": "Post-Sale / Trade Discount Documentation",
        "description": "Unusually high trade discount applied without standard commercial rationale.",
        "risk_impact": "Under Section 15(3), discounts are deductible from taxable value only if agreed upon before supply and linked to relevant invoices.",
        "actions": [
            {
                "priority": "HIGH",
                "action": "Verify Pre-Supply Discount Agreement",
                "detail": "Ensure terms of the discount scheme were documented in the commercial contract prior to supply.",
                "doc_required": "Vendor Commercial Scheme Circular"
            },
            {
                "priority": "MEDIUM",
                "action": "Ensure Proportionate ITC Reversal if Applicable",
                "detail": "If discount was passed post-supply via credit note, confirm ITC reversal in GSTR-3B Table 4(B)(2).",
                "doc_required": "GSTR-3B Return Copy"
            }
        ]
    }
}


DEFAULT_ACTIONS = {
    "title": "Standard Compliance Verification",
    "description": "Routine compliance checklist for MSME GST audit readiness.",
    "risk_impact": "Low compliance risk. Maintain documentation for statutory 6-year retention period under GST Section 36.",
    "actions": [
        {
            "priority": "LOW",
            "action": "Match with Monthly GSTR-2B Statement",
            "detail": "Ensure invoice details appear accurately in your automated GSTR-2B ITC statement.",
            "doc_required": "GSTR-2B Entry"
        },
        {
            "priority": "LOW",
            "action": "Confirm Payment within 180 Days",
            "detail": "Ensure supplier invoice payment is released within statutory 180-day timeline to avoid ITC reversal.",
            "doc_required": "Accounts Payable Ledger"
        }
    ]
}


def get_recommendations_for_invoice(invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate customized compliance action checklist for an invoice.
    """
    anomaly_type = invoice_data.get("anomaly_type", "Normal")
    risk_level = invoice_data.get("risk_level", "Low Risk")

    rec = ANOMALY_ACTION_MAP.get(anomaly_type, DEFAULT_ACTIONS)

    # Determine recommended workflow action
    if risk_level == "High Risk":
        workflow_action = "HOLD_ITC_AND_INVESTIGATE"
        workflow_label = "Hold ITC & Escalate for Audit"
        workflow_badge = "danger"
    elif risk_level == "Medium Risk":
        workflow_action = "VERIFY_DOCUMENTS"
        workflow_label = "Request Supporting Documents"
        workflow_badge = "warning"
    else:
        workflow_action = "PROCEED_TO_FILE"
        workflow_label = "Safe to Claim in GSTR-3B"
        workflow_badge = "success"

    return {
        "title": rec["title"],
        "description": rec["description"],
        "risk_impact": rec["risk_impact"],
        "workflow_action": workflow_action,
        "workflow_label": workflow_label,
        "workflow_badge": workflow_badge,
        "actions": rec["actions"]
    }

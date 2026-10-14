"""
routes/report_routes.py
=======================
Compliance reporting, ML model performance metrics, and CSV export functionality.
"""

import os
import json
import io
import pandas as pd
from flask import Blueprint, render_template, jsonify, send_file, request
from database.database import SessionLocal, Invoice, Analysis

report_bp = Blueprint("reports", __name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_JSON_PATH = os.path.join(ROOT_DIR, "results", "model_evaluation_metrics.json")


@report_bp.route("/reports")
def reports_view():
    """Render the invoice CSV export page."""
    return render_template("export_reports.html")


@report_bp.route("/api/model-performance", methods=["GET"])
def api_model_performance():
    """
    GET /api/model-performance
    Returns accuracy, precision, recall, F1-scores, and per-anomaly-type detection
    for Isolation Forest, LOF, and Z-Score.
    """
    try:
        if not os.path.exists(METRICS_JSON_PATH):
            return jsonify({
                "status": "error",
                "message": "Model performance metrics not found. Run models/evaluate_models.py first."
            }), 404

        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return jsonify({
            "status": "success",
            "data": data
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@report_bp.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    """
    GET /api/export/csv
    Download compliance report as CSV file.
    Supports filtering by risk_level (e.g. ?risk_level=High%20Risk).
    """
    session = SessionLocal()
    try:
        risk_level = request.args.get("risk_level", "").strip()

        query = session.query(
            Invoice.invoice_no,
            Invoice.invoice_date,
            Invoice.supplier_id,
            Invoice.supplier_gstin,
            Invoice.buyer_id,
            Invoice.buyer_gstin,
            Invoice.product_category,
            Invoice.quantity,
            Invoice.unit_price,
            Invoice.taxable_value,
            Invoice.gst_rate,
            Invoice.total_tax,
            Invoice.total_amount,
            Analysis.isolation_score,
            Analysis.lof_score,
            Analysis.max_abs_zscore,
            Analysis.risk_score,
            Analysis.risk_level,
            Analysis.anomaly_type,
            Analysis.explanation
        ).join(Analysis, Invoice.invoice_no == Analysis.invoice_no)

        if risk_level:
            query = query.filter(Analysis.risk_level == risk_level)

        results = query.order_by(Analysis.risk_score.desc()).all()

        # Build DataFrame
        columns = [
            "Invoice No", "Date", "Supplier ID", "Supplier GSTIN", "Buyer ID", "Buyer GSTIN",
            "Category", "Quantity", "Unit Price", "Taxable Value", "GST Rate", "Total Tax",
            "Total Amount", "Isolation Forest Score", "LOF Score", "Max Z-Score",
            "Risk Score", "Risk Level", "Anomaly Type", "Explanation"
        ]
        df = pd.DataFrame(results, columns=columns)

        # Write to in-memory buffer
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding="utf-8")
        output.seek(0)

        filename = f"gst_compliance_report_{risk_level.lower().replace(' ', '_') or 'all'}.csv"
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        session.close()

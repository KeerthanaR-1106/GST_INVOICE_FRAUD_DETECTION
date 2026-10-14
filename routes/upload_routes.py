"""
routes/upload_routes.py
=======================
Handles CSV upload, validation, full ML inference pipeline execution,
and SQLite database insertion:
  User uploads CSV
  ↓
  Validate CSV (format, columns, non-empty)
  ↓
  Preprocess & generate 43 ML features
  ↓
  Run Isolation Forest inference
  ↓
  Run Local Outlier Factor inference
  ↓
  Run Statistical Z-Score inference
  ↓
  Compute Ensemble Risk Score & Risk Level
  ↓
  Generate compliance explanations
  ↓
  Insert transactions & analysis into SQLite database
  ↓
  Return status & metrics to UI
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from config import Config
from database.database import DB_PATH, SessionLocal, Invoice, Analysis
from services.anomaly_service import run_anomaly_pipeline

upload_bp = Blueprint("upload", __name__)

REQUIRED_CSV_COLUMNS = [
    "invoice_no", "invoice_date", "supplier_id", "supplier_gstin",
    "buyer_id", "buyer_gstin", "product_category", "hsn_code",
    "quantity", "unit_price", "discount_percentage", "discount_amount",
    "taxable_value", "gst_rate", "cgst", "sgst", "igst",
    "cess_rate", "cess_amount", "total_tax", "total_amount",
    "payment_method"
]


@upload_bp.route("/upload")
def upload_view():
    """Render file upload page."""
    return render_template("upload.html")


@upload_bp.route("/api/upload", methods=["POST"])
def api_upload_csv():
    """
    POST /api/upload
    Receives multipart/form-data containing 'file'.
    Executes complete end-to-end anomaly detection pipeline and stores results in SQLite.
    """
    # 1. Validate file presence
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file part in request. Please select a valid CSV file."
        }), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "No file selected. Please choose a CSV file to upload."
        }), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Only CSV (.csv) files are supported."
        }), 400

    # 2. Save file temporarily
    filename = secure_filename(file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(temp_path)

    try:
        # 3. Read and validate structure
        try:
            df_raw = pd.read_csv(temp_path)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Malformed CSV file could not be parsed: {str(e)}"
            }), 400

        if df_raw.empty:
            return jsonify({
                "status": "error",
                "message": "The uploaded CSV file is empty. Please upload a file containing invoice data."
            }), 400

        # Validate required columns
        missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in df_raw.columns]
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Missing required columns ({len(missing_cols)}): {', '.join(missing_cols)}"
            }), 400

        # Handle numeric validation and date parsing
        df_raw["invoice_date"] = pd.to_datetime(df_raw["invoice_date"], errors="coerce")
        if df_raw["invoice_date"].isnull().any():
            median_date = df_raw["invoice_date"].dropna().median()
            df_raw["invoice_date"] = df_raw["invoice_date"].fillna(median_date)

        num_cols = ["quantity", "unit_price", "discount_percentage", "discount_amount",
                    "taxable_value", "gst_rate", "cgst", "sgst", "igst",
                    "cess_rate", "cess_amount", "total_tax", "total_amount"]
        for c in num_cols:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0).clip(lower=0.0)

        # 4. End-to-End ML Pipeline Execution via Anomaly Service
        n_samples = len(df_raw)
        analysis_records = run_anomaly_pipeline(df_raw)
        risk_levels = analysis_records["risk_level"].tolist()
        risk_score = analysis_records["risk_score"].values
        df_raw["invoice_date"] = pd.to_datetime(df_raw["invoice_date"]).dt.strftime("%Y-%m-%d")


        # Insert or update in SQLite
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            # Replace existing invoices with same invoice_no
            inv_tuples = [tuple(x) for x in df_raw[REQUIRED_CSV_COLUMNS].to_numpy()]
            cursor.executemany(f"""
                INSERT OR REPLACE INTO invoices ({', '.join(REQUIRED_CSV_COLUMNS)})
                VALUES ({', '.join(['?']*len(REQUIRED_CSV_COLUMNS))});
            """, inv_tuples)

            ans_cols = list(analysis_records.columns)
            ans_tuples = [tuple(x) for x in analysis_records.to_numpy()]
            cursor.executemany(f"""
                INSERT OR REPLACE INTO analysis ({', '.join(ans_cols)})
                VALUES ({', '.join(['?']*len(ans_cols))});
            """, ans_tuples)

            conn.commit()
        finally:
            conn.close()

        # Summary counts
        lvl_series = pd.Series(risk_levels)
        high_cnt = int((lvl_series == "High Risk").sum())
        med_cnt = int((lvl_series == "Medium Risk").sum())
        low_cnt = int((lvl_series == "Low Risk").sum())

        return jsonify({
            "status": "success",
            "message": f"Successfully processed {n_samples:,} invoices and updated SQLite compliance database.",
            "processed_count": n_samples,
            "summary": {
                "high_risk": high_cnt,
                "medium_risk": med_cnt,
                "low_risk": low_cnt,
                "suspicious": high_cnt + med_cnt,
                "avg_risk_score": round(float(np.mean(risk_score)), 2)
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Internal pipeline execution error: {str(e)}"
        }), 500
    finally:
        # Cleanup temporary uploaded file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@upload_bp.route("/api/demo-analysis", methods=["POST"])
def api_demo_analysis():
    """
    POST /api/demo-analysis
    Executes live pipeline on packaged sample dataset for hackathon judges
    without requiring manual file upload.
    """
    try:
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gst_invoices.csv")
        if not os.path.exists(sample_path):
            return jsonify({"status": "error", "message": "Demo data file not found."}), 404

        df_raw = pd.read_csv(sample_path).head(100)
        n_samples = len(df_raw)

        # Run pipeline
        analysis_records = run_anomaly_pipeline(df_raw)
        risk_levels = analysis_records["risk_level"].tolist()
        risk_score = analysis_records["risk_score"].values
        df_raw["invoice_date"] = pd.to_datetime(df_raw["invoice_date"]).dt.strftime("%Y-%m-%d")

        # Insert / Update SQLite
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            inv_tuples = [tuple(x) for x in df_raw[REQUIRED_CSV_COLUMNS].to_numpy()]
            cursor.executemany(f"""
                INSERT OR REPLACE INTO invoices ({', '.join(REQUIRED_CSV_COLUMNS)})
                VALUES ({', '.join(['?']*len(REQUIRED_CSV_COLUMNS))});
            """, inv_tuples)

            ans_cols = list(analysis_records.columns)
            ans_tuples = [tuple(x) for x in analysis_records.to_numpy()]
            cursor.executemany(f"""
                INSERT OR REPLACE INTO analysis ({', '.join(ans_cols)})
                VALUES ({', '.join(['?']*len(ans_cols))});
            """, ans_tuples)

            conn.commit()
        finally:
            conn.close()

        lvl_series = pd.Series(risk_levels)
        high_cnt = int((lvl_series == "High Risk").sum())
        med_cnt = int((lvl_series == "Medium Risk").sum())
        low_cnt = int((lvl_series == "Low Risk").sum())

        return jsonify({
            "status": "success",
            "message": f"Successfully evaluated {n_samples} demo invoices through all 3 unsupervised models.",
            "processed_count": n_samples,
            "summary": {
                "high_risk": high_cnt,
                "medium_risk": med_cnt,
                "low_risk": low_cnt,
                "suspicious": high_cnt + med_cnt,
                "avg_risk_score": round(float(np.mean(risk_score)), 2)
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


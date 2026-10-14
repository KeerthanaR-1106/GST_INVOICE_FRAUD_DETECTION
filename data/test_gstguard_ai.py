"""
data/test_gstguard_ai.py
========================
Comprehensive Verification Suite for GSTGuard AI Transformation:
  1. Blueprint and App Factory Registration
  2. Database Schema & Investigation Model
  3. Investigation Workflow API (GET, PATCH, status updates)
  4. Supplier Risk Intelligence & APIs
  5. Priority Review Queue & APIs
  6. Compliance Copilot Q&A Engine & API
  7. Explanation Service & Empirical Evidence Benchmarks
  8. MSME Compliance Action Center Checklists
  9. Report Generation Service & Views
  10. Centralized Anomaly Pipeline & Demo Analysis
"""

import unittest
import json
from app import create_app
from database.database import (
    SessionLocal, Invoice, Analysis, Investigation,
    get_dashboard_summary, get_review_queue, get_supplier_metrics,
    get_supplier_detail, get_or_create_investigation, update_investigation
)
from services.explanation_service import get_detailed_explanation, build_evidence_items
from services.recommendation_service import get_recommendations_for_invoice
from services.copilot_service import query_copilot
from services.report_service import generate_invoice_report_data
from services.anomaly_service import run_anomaly_pipeline
import pandas as pd


class TestGSTGuardAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    # ── 1. App Factory & Blueprints ──────────────────────────────────────────
    def test_01_blueprints_registered(self):
        """All 6 core blueprints must be registered."""
        blueprint_names = set(self.app.blueprints.keys())
        expected = {"dashboard", "invoices", "reports", "upload", "suppliers", "copilot"}
        self.assertTrue(expected.issubset(blueprint_names), f"Missing blueprints: {expected - blueprint_names}")

    # ── 2. Database Schema & Models ──────────────────────────────────────────
    def test_02_database_models(self):
        """Database must have Invoices, Analysis, and Investigations accessible."""
        session = SessionLocal()
        try:
            inv_count = session.query(Invoice).count()
            ans_count = session.query(Analysis).count()
            self.assertGreater(inv_count, 0)
            self.assertGreater(ans_count, 0)

            # Test investigation query
            first_inv = session.query(Invoice).first()
            self.assertIsNotNone(first_inv)

            inv_rec = get_or_create_investigation(first_inv.invoice_no)
            self.assertIn("status", inv_rec)
            self.assertIn("invoice_no", inv_rec)
        finally:
            session.close()

    # ── 3. Investigation Workflow ────────────────────────────────────────────
    def test_03_investigation_workflow_api(self):
        """Test GET and PATCH /api/investigations/<invoice_no>."""
        session = SessionLocal()
        try:
            first_inv = session.query(Invoice).first()
            inv_no = first_inv.invoice_no
        finally:
            session.close()

        # GET
        res_get = self.client.get(f"/api/investigations/{inv_no}")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.get_json()
        self.assertEqual(data_get["status"], "success")

        # PATCH update
        payload = {
            "status": "UNDER REVIEW",
            "notes": "Verified against e-Way bill #EW12345.",
            "assigned_to": "Senior Officer Raman"
        }
        res_patch = self.client.patch(
            f"/api/investigations/{inv_no}",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res_patch.status_code, 200)
        data_patch = res_patch.get_json()
        self.assertEqual(data_patch["data"]["status"], "UNDER REVIEW")
        self.assertEqual(data_patch["data"]["assigned_to"], "Senior Officer Raman")

    # ── 4. Supplier Intelligence ─────────────────────────────────────────────
    def test_04_supplier_service_and_apis(self):
        """Test supplier aggregation queries, views, and APIs."""
        suppliers_res = get_supplier_metrics(page=1, per_page=10)
        self.assertIn("items", suppliers_res)
        self.assertGreater(len(suppliers_res["items"]), 0)

        first_supplier = suppliers_res["items"][0]
        self.assertIn("supplier_risk_score", first_supplier)
        self.assertIn("supplier_risk_level", first_supplier)

        # Directory and scorecard must aggregate all GSTIN registrations under one supplier ID.
        multi_gstin_supplier = "SUP0079"
        directory_supplier = next(
            item for item in get_supplier_metrics(page=1, per_page=1000)["items"]
            if item["supplier_id"] == multi_gstin_supplier
        )
        scorecard_supplier = get_supplier_detail(multi_gstin_supplier)
        self.assertEqual(directory_supplier["supplier_risk_score"], scorecard_supplier["supplier_risk_score"])
        self.assertEqual(directory_supplier["supplier_risk_level"], scorecard_supplier["supplier_risk_level"])

        # Test API
        api_res = self.client.get("/api/suppliers")
        self.assertEqual(api_res.status_code, 200)
        api_data = api_res.get_json()
        self.assertEqual(api_data["status"], "success")

        # Test detail view
        sup_id = first_supplier["supplier_id"]
        detail_res = self.client.get(f"/suppliers/{sup_id}")
        self.assertEqual(detail_res.status_code, 200)
        self.assertIn(b"Supplier Risk Score", detail_res.data)

    # ── 5. Priority Review Queue ─────────────────────────────────────────────
    def test_05_priority_review_queue(self):
        """Test ranked review queue generation and view."""
        queue = get_review_queue(limit=25)
        self.assertIsInstance(queue, list)
        self.assertGreater(len(queue), 0)

        top_item = queue[0]
        self.assertIn("priority_score", top_item)
        self.assertIn("consensus_count", top_item)
        self.assertGreaterEqual(top_item["priority_score"], 0)

        # Review Queue Route
        res = self.client.get("/review-queue")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Priority Review Queue", res.data)

        # Review Queue API
        api_res = self.client.get("/api/review-queue")
        self.assertEqual(api_res.status_code, 200)

    # ── 6. Compliance Copilot ────────────────────────────────────────────────
    def test_06_compliance_copilot(self):
        """Test deterministic Copilot query engine."""
        # General query
        res1 = query_copilot("Explain GSTR-2B rules")
        self.assertIn("answer", res1)
        self.assertIn("GSTR-2B", res1["answer"])

        # General query API accepts the null invoice number sent by the web client.
        general_api_res = self.client.post(
            "/api/copilot/query",
            data=json.dumps({"query": "Explain GSTR-2B compliance rules", "invoice_no": None}),
            content_type="application/json"
        )
        self.assertEqual(general_api_res.status_code, 200)
        general_data = general_api_res.get_json()
        self.assertEqual(general_data["status"], "success")
        self.assertIn("GSTR-2B", general_data["data"]["answer"])

        risk_api_res = self.client.post(
            "/api/copilot/query",
            data=json.dumps({"query": "What are the key risk categories?", "invoice_no": None}),
            content_type="application/json"
        )
        self.assertEqual(risk_api_res.status_code, 200)
        self.assertEqual(risk_api_res.get_json()["data"]["category"], "risk_thresholds")
        self.assertIn("40–69", risk_api_res.get_json()["data"]["answer"])

        # Invoice query
        session = SessionLocal()
        try:
            high_risk_inv = session.query(Invoice, Analysis).join(
                Analysis, Invoice.invoice_no == Analysis.invoice_no
            ).filter(Analysis.risk_level == "High Risk").first()
            if high_risk_inv:
                inv_dict = high_risk_inv[0].to_dict()
                inv_dict.update(high_risk_inv[1].to_dict())
            else:
                inv_dict = None
        finally:
            session.close()

        if inv_dict:
            res_why = query_copilot("Why was this invoice flagged?", invoice_data=inv_dict)
            self.assertIn("answer", res_why)
            self.assertIn("Risk Score", res_why["answer"])

            embedded_invoice_api_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": f"Why was invoice {inv_dict['invoice_no']} flagged?"
                }),
                content_type="application/json"
            )
            self.assertEqual(embedded_invoice_api_res.status_code, 200)
            embedded_data = embedded_invoice_api_res.get_json()["data"]
            self.assertEqual(embedded_data["category"], "explanation")
            self.assertIn(inv_dict["invoice_no"], embedded_data["answer"])

            supplier_summary_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": (
                        f"What company provided invoice {inv_dict['invoice_no']}, how many invoices "
                        "has it produced, and what are their risk levels and anomaly patterns?"
                    )
                }),
                content_type="application/json"
            )
            self.assertEqual(supplier_summary_res.status_code, 200)
            supplier_summary = supplier_summary_res.get_json()["data"]
            self.assertEqual(supplier_summary["category"], "supplier_summary")
            self.assertIn("Invoices in this dataset", supplier_summary["answer"])
            self.assertIn("Anomaly patterns", supplier_summary["answer"])
            self.assertIn("Not recorded in this dataset", supplier_summary["answer"])

            risk_reduction_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": f"How can I reduce the risk score of invoice {inv_dict['invoice_no']}?"
                }),
                content_type="application/json"
            )
            self.assertEqual(risk_reduction_res.status_code, 200)
            risk_reduction = risk_reduction_res.get_json()["data"]
            self.assertEqual(risk_reduction["category"], "risk_reduction")
            self.assertIn("rerun the analysis", risk_reduction["answer"])

            # Copilot API
            api_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({"query": "What is the tax discrepancy?", "invoice_no": inv_dict["invoice_no"]}),
                content_type="application/json"
            )
            self.assertEqual(api_res.status_code, 200)
            data = api_res.get_json()
            self.assertEqual(data["status"], "success")

            model_api_res = self.client.post(
                "/api/copilot/query",
                data=json.dumps({
                    "query": "What models detected this anomaly?",
                    "invoice_no": inv_dict["invoice_no"]
                }),
                content_type="application/json"
            )
            self.assertEqual(model_api_res.status_code, 200)
            self.assertEqual(model_api_res.get_json()["data"]["category"], "models")

        unknown_invoice_res = self.client.post(
            "/api/copilot/query",
            data=json.dumps({"query": "Why was invoice INV999999 flagged?"}),
            content_type="application/json"
        )
        self.assertEqual(unknown_invoice_res.status_code, 404)

    # ── 7. AI Explanation & Empirical Evidence ───────────────────────────────
    def test_07_explanation_and_evidence(self):
        """Test evidence extraction and actual vs typical benchmarks."""
        dummy_invoice = {
            "invoice_no": "TEST-001",
            "quantity": 1200.0,
            "unit_price": 95000.0,
            "discount_percentage": 65.0,
            "taxable_value": 250000.0,
            "gst_rate": 18.0,
            "total_tax": 45000.0,
            "cgst": 22500.0,
            "sgst": 22500.0,
            "igst": 0.0,
            "total_amount": 295000.0,
            "isolation_anomaly": 1,
            "isolation_score": 0.65,
            "lof_anomaly": 1,
            "lof_score": 2.1,
            "zscore_anomaly": 1,
            "max_abs_zscore": 4.5,
            "risk_score": 88.5,
            "risk_level": "High Risk",
            "anomaly_type": "Extreme Unit Price"
        }
        exp = get_detailed_explanation(dummy_invoice)
        self.assertEqual(exp["risk_level"], "High Risk")
        self.assertEqual(exp["consensus_count"], 3)
        self.assertGreater(exp["flagged_evidence_count"], 0)

        # Check evidence features
        features = [e["feature"] for e in exp["evidence"]]
        self.assertIn("Unit Price", features)
        self.assertIn("Billed Quantity", features)

    # ── 8. MSME Recommendation Action Center ─────────────────────────────────
    def test_08_msme_recommendations(self):
        """Test actionable checklists generated per anomaly type."""
        dummy_tax_mismatch = {
            "anomaly_type": "Tax Calculation Mismatch",
            "risk_level": "High Risk"
        }
        rec = get_recommendations_for_invoice(dummy_tax_mismatch)
        self.assertEqual(rec["workflow_action"], "HOLD_ITC_AND_INVESTIGATE")
        self.assertGreater(len(rec["actions"]), 0)
        self.assertTrue(any("GSTR-1" in a["action"] or "ITC" in a["action"] for a in rec["actions"]))

    # ── 9. Compliance Report Generation ──────────────────────────────────────
    def test_09_compliance_report(self):
        """Test report data generation and report view."""
        session = SessionLocal()
        try:
            first_inv = session.query(Invoice, Analysis).join(
                Analysis, Invoice.invoice_no == Analysis.invoice_no
            ).first()
            inv_dict = first_inv[0].to_dict()
            inv_dict.update(first_inv[1].to_dict())
            inv_no = inv_dict["invoice_no"]
        finally:
            session.close()

        rep = generate_invoice_report_data(inv_dict)
        self.assertIn("report_id", rep)
        self.assertIn("recommendations", rep)
        self.assertIn("explanation", rep)

        # View route
        res = self.client.get(f"/invoice/{inv_no}/report")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"GSTGuard AI Compliance Report", res.data)

        # API route
        api_res = self.client.get(f"/api/invoices/{inv_no}/report")
        self.assertEqual(api_res.status_code, 200)

    # ── 10. Centralized Anomaly Pipeline & Demo Analysis ─────────────────────
    def test_10_centralized_pipeline_and_demo(self):
        """Test anomaly service execution and demo analysis endpoint."""
        demo_res = self.client.post("/api/demo-analysis")
        self.assertEqual(demo_res.status_code, 200)
        demo_data = demo_res.get_json()
        self.assertEqual(demo_data["status"], "success")
        self.assertEqual(demo_data["processed_count"], 100)
        self.assertIn("summary", demo_data)

    # ── 11. Dashboard & Invoice Details Views ────────────────────────────────
    def test_11_dashboard_and_details_views(self):
        """Test dashboard renders with review queue and invoice details with Risk Story."""
        dash_res = self.client.get("/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn(b"Priority Review Queue", dash_res.data)
        self.assertIn(b"Top Risky Suppliers", dash_res.data)

        session = SessionLocal()
        try:
            first_inv = session.query(Invoice).first()
            inv_no = first_inv.invoice_no
        finally:
            session.close()

        det_res = self.client.get(f"/invoice/{inv_no}")
        self.assertEqual(det_res.status_code, 200)
        self.assertIn(b"Empirical Evidence vs. MSME Benchmarks", det_res.data)
        self.assertIn(b"MSME Compliance Action Center", det_res.data)
        self.assertIn(b"Compliance Investigation Tracker", det_res.data)


if __name__ == "__main__":
    unittest.main()

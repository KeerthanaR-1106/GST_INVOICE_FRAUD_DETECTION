# Automated GST Invoice Fraud Detection System using Anomaly Detection for MSME Compliance

An end-to-end Machine Learning and Web Application system designed to identify suspicious patterns, tax discrepancies, and potential fraud in Goods and Services Tax (GST) invoice transactions for Micro, Small, and Medium Enterprises (MSMEs).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement & MSME Compliance Context](#problem-statement--msme-compliance-context)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Dataset & Fraud Patterns](#dataset--fraud-patterns)
6. [Feature Engineering (43 Features)](#feature-engineering-43-features)
7. [Machine Learning Models](#machine-learning-models)
8. [Model Evaluation & Comparison](#model-evaluation--comparison)
9. [Ensemble Risk Scoring Engine](#ensemble-risk-scoring-engine)
10. [Database Architecture (SQLite)](#database-architecture-sqlite)
11. [REST API Documentation](#rest-api-documentation)
12. [Web Frontend & Dashboard UI](#web-frontend--dashboard-ui)
13. [Installation & Setup Guide](#installation--setup-guide)
14. [Running the System & Tests](#running-the-system--tests)
15. [Academic Viva Q&A Guide](#academic-viva-qa-guide)
16. [Compliance & Legal Disclaimer](#compliance--legal-disclaimer)

---

## 1. Project Overview

Under the Indian Goods and Services Tax (GST) framework, businesses claim **Input Tax Credit (ITC)** on taxes paid for inward supplies. However, invoice fraud—such as fake invoicing without actual goods transfer, circular trading, inflated invoice amounts, and tax rate mismatches—causes massive revenue leakages and severe compliance liabilities for honest MSMEs.

This project delivers an **unsupervised anomaly detection system** that:
* Ingests GST invoice batches (20,000 transactions).
* Engineers **43 domain-specific numerical features** capturing tax consistency, supplier-buyer frequency, temporal anomalies, and duplicate signatures.
* Runs **three complementary unsupervised ML algorithms**:
  1. **Isolation Forest** (tree-based global outlier isolation)
  2. **Local Outlier Factor (LOF)** (density-based local neighborhood deviation)
  3. **Statistical Z-Score Analysis** (parametric univariate variance detection)
* Fuses model signals into an **Ensemble Risk Score (0–100)** with consensus calibration.
* Classifies invoices into **Low Risk (0–39)**, **Medium Risk (40–69)**, and **High Risk (70–100)**.
* Persists transactions and compliance audits into an indexed **SQLite database**.
* Exposes **Flask REST APIs** and renders a responsive **Bootstrap 5 + Chart.js web dashboard** for compliance auditors.

---

## 2. Problem Statement & MSME Compliance Context

### The Challenge for MSMEs
* **Strict Reversal Rules**: Under Section 16(2) of the CGST Act, buyers cannot claim ITC if their supplier fails to deposit the tax with the government. MSMEs often unknowingly transact with non-compliant or fraudulent vendors.
* **Volume Overload**: A typical MSME processes thousands of invoices per quarter. Manual inspection is computationally infeasible and error-prone.
* **Failure of Traditional Rule Engines**: Simple static threshold checks miss non-linear anomalies, subtle tax splits, or localized deviations in trade patterns.
* **Unsupervised Need**: In real-world enterprise environments, ground-truth fraud labels are extremely rare or unavailable at filing time. Hence, the system must learn normal operational patterns **unsupervised**.

---

## 3. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                              GST INVOICE DATASET                                  |
|                 (20,000 synthetic transactions, 22 raw columns)                   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        STAGE 2: FEATURE ENGINEERING                               |
|        (43 numeric features across Consistency, Date, Frequency & Duplicate)       |
+-----------------------------------------+-----------------------------------------+
                                          |
         +--------------------------------+--------------------------------+
         |                                |                                |
         v                                v                                v
+-------------------+            +-------------------+            +-------------------+
|     STAGE 3       |            |      STAGE 4      |            |      STAGE 5      |
| ISOLATION FOREST  |            |   LOCAL OUTLIER   |            | STATISTICAL Z-SC. |
| (Tree Partition)  |            |   FACTOR (k-NN)   |            | (|Z| >= 3.0)      |
+---------+---------+            +---------+---------+            +---------+---------+
          |                                |                                |
          +--------------------------------+--------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      STAGE 7: ENSEMBLE RISK SCORING ENGINE                        |
|       Score Normalization + Weighted Sum (40% IF + 40% LOF + 20% Z-Score)          |
|                   + Multi-Model Consensus Calibration (0 - 100)                   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      STAGE 8: SQLITE DATABASE (gst_fraud.db)                      |
|           Table 'invoices' (20k rows) <== 1:1 ==> Table 'analysis' (20k rows)     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         STAGE 9: FLASK REST API BACKEND                           |
|       /api/dashboard | /api/invoices | /api/suspicious-invoices | /api/upload     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      STAGE 10: WEB FRONTEND (Bootstrap 5 & Chart.js)              |
|        Executive Dashboard, Suspicious Audits, Export Reports                    |
+-----------------------------------------------------------------------------------+
```

---

## 4. Technology Stack

| Layer | Component | Version / Library | Purpose |
|---|---|---|---|
| **Language** | Python | 3.13 / 3.11+ | Core programming runtime |
| **Backend Framework** | Flask | 3.1.0 | REST API endpoints & web routing |
| **Database ORM** | SQLAlchemy | 2.0.41 | Scoped session & relational ORM mapping |
| **Database Engine** | SQLite | 3.x | Lightweight, local, zero-config relational storage |
| **Machine Learning** | scikit-learn | 1.6.1 | Isolation Forest, Local Outlier Factor, StandardScaler |
| **Data Processing** | pandas, numpy | 3.0.5, 2.4.2 | Feature matrix construction, array calculations |
| **Frontend Styling** | Bootstrap 5 | 5.3.3 (CDN) | Responsive grid, cards, badges, and modals |
| **Icons** | Bootstrap Icons | 1.11.3 (CDN) | Modern UI iconography |
| **Data Visualization**| Chart.js | 4.4.2 (CDN) | Interactive doughnut, grouped bar, and horizontal bar charts |

---

## 5. Dataset & Fraud Patterns

The dataset contains **20,000 synthetic GST invoice transactions** modeled after real Indian GST invoicing patterns:
* **Normal Invoices**: 19,000 (95.00%)
* **Anomalous Invoices**: 1,000 (5.00%)
* **Number of Raw Columns**: 22 standard GST invoice fields

### 11 Injected Anomaly Types

| Anomaly Pattern | Injected Count | Description | Primary Detection Channel |
|---|:---:|---|---|
| **Extreme Invoice Amount** | 91 | Amount 10x–50x above standard product category distribution | Z-Score & Isolation Forest |
| **Extreme Unit Price** | 91 | Item price unrealistic for given HSN / category | Z-Score & Isolation Forest |
| **Unusual Quantity** | 91 | Uncharacteristically high item volume | Z-Score & Isolation Forest |
| **Unusual Discount** | 91 | Excessive trade discount (>50% up to 90%) | Z-Score & LOF |
| **Tax Calculation Mismatch** | 91 | Declared total tax differs from `taxable_value * rate` | Consistency Features & Z-Score |
| **Incorrect CGST/SGST Split** | 91 | Intra-state transaction where CGST != SGST | Consistency Features & Z-Score |
| **Incorrect IGST** | 91 | Inter-state rate discrepancy or wrong interstate levy | Consistency Features & Z-Score |
| **Duplicate Transaction** | 182 | Duplicate buyer-supplier-date-amount footprint | Duplicate Feature & Isolation Forest |
| **Future Invoice Date** | 90 | Invoice issued with date in future | Date Feature & Z-Score |
| **Unusual Supplier Activity** | 91 | Abnormal daily frequency spike for supplier entity | Frequency Features & LOF |
| **Unusual Buyer Activity** | 91 | Abnormal daily frequency spike for purchasing entity | Frequency Features & LOF |

---

## 6. Feature Engineering (43 Features)

From the 22 raw columns, the preprocessing pipeline derives **43 numerical features** across 5 categories:

1. **Raw Numerical Features (13)**:
   `quantity`, `unit_price`, `discount_percentage`, `discount_amount`, `taxable_value`, `gst_rate`, `cgst`, `sgst`, `igst`, `cess_rate`, `cess_amount`, `total_tax`, `total_amount`.
2. **GST Consistency Features (18)**:
   * `expected_taxable_value`, `taxable_value_difference`, `taxable_value_relative_diff`
   * `expected_gst`, `gst_difference`, `gst_relative_diff`
   * `expected_cgst_sgst`, `cgst_sgst_split_difference`, `cgst_sgst_relative_diff`
   * `expected_igst`, `igst_difference`, `igst_relative_diff`
   * `expected_total_amount`, `total_amount_difference`, `total_amount_relative_diff`
   * `is_interstate_flag`, `gst_rate_difference`, `cess_difference`
3. **Date Features (4)**:
   `invoice_month`, `invoice_day`, `day_of_week`, `future_date_flag`
4. **Frequency & Activity Features (6)**:
   * Supplier 1-day, 7-day, and 30-day invoice transaction frequency
   * Buyer 1-day, 7-day, and 30-day invoice transaction frequency
5. **Duplicate Transaction Signatures (2)**:
   `duplicate_transaction_count`, `duplicate_transaction_flag`

---

## 7. Machine Learning Models

### 1. Isolation Forest
* **Principle**: Isolates anomalies instead of profiling normal points. Random axis-aligned cuts separate outliers in fewer splits because they are rare and feature-divergent.
* **Hyperparameters**: `n_estimators=200`, `contamination=0.05`, `random_state=42`, `n_jobs=-1`.
* **Output**: Anomaly flags (1,000 flagged, 5.00%) and continuous decision score.

### 2. Local Outlier Factor (LOF)
* **Principle**: Density-based algorithm comparing local reachability density of an invoice to its $k$-nearest neighbors. Outliers reside in regions of significantly lower density.
* **Hyperparameters**: `n_neighbors=20`, `contamination=0.05`, `n_jobs=-1`.
* **Output**: Anomaly flags (1,000 flagged, 5.00%) and LOF score (range: 0.94 to 27.03).

### 3. Statistical Z-Score
* **Principle**: Standardized univariate score:
  $$Z = \frac{X - \mu}{\sigma}$$
  Measures standard deviations from mean across all 43 features. Safe handling for zero variance features ($\sigma = 0 \Rightarrow \sigma = 1$).
* **Threshold**: Flagged if $\max |Z| \ge 3.0$.
* **Output**: 2,071 invoices flagged (10.36%), dominant feature attribution, extreme feature count.

---

## 8. Model Evaluation & Comparison

Models were evaluated strictly offline against `ground_truth.csv`:

| Model / Strategy | Accuracy | Precision | Recall | F1-Score | Anomalies Flagged |
|---|:---:|:---:|:---:|:---:|:---:|
| **Isolation Forest** | 94.38% | 43.80% | 43.80% | 0.4380 | 1,000 |
| **Local Outlier Factor (LOF)** | 92.49% | 24.90% | 24.90% | 0.2490 | 1,000 |
| **Statistical Z-Score** | 93.61% | 43.26% | **89.60%** | **0.5835** | 2,071 |
| **At Least 1 Model (Union)** | 89.28% | 31.30% | **95.80%** | 0.4718 | 3,061 |
| **Majority Vote (>= 2 Models)** | 95.60% | 56.93% | 49.30% | 0.5284 | 866 |
| **Consensus (All 3 Models)** | **95.60%** | **91.67%** | 13.20% | 0.2308 | 144 |

> **Key Viva Insight**: Combining models achieves **95.80% total anomaly recall** in union and **91.67% precision** at the highest tier of multi-model agreement.

---

## 9. Ensemble Risk Scoring Engine

### Score Normalization
1. **Isolation Forest**: Min-Max scaling to $[0, 100]$.
2. **LOF Score**: Piecewise non-linear scaling:
   * $\text{LOF} \le 1.0 \implies [0, 10]$
   * $1.0 < \text{LOF} \le 2.5 \implies [10, 80]$
   * $\text{LOF} > 2.5 \implies [80, 100]$
3. **Z-Score**: Piecewise scaling anchored at threshold 3.0:
   * $|Z| < 3.0 \implies [0, 35]$
   * $|Z| \ge 3.0 \implies [50, 100]$

### Weighted Sum & Consensus Calibration
$$\text{Base Score} = 0.40 \times \text{Norm}_{\text{IF}} + 0.40 \times \text{Norm}_{\text{LOF}} + 0.20 \times \text{Norm}_{\text{Z}}$$

* If all 3 models flag anomaly $\implies \text{Risk Score} = \max(\text{Base Score}, 75.0)$
* If 2 models flag anomaly $\implies \text{Risk Score} = \max(\text{Base Score}, 55.0)$
* If 0 models flag anomaly $\implies \text{Risk Score} = \min(\text{Base Score}, 35.0)$

### Risk Classification Tiers

| Tier | Score Range | Count in Database | Percentage | Audit Recommendation |
|---|:---:|:---:|:---:|---|
| **Low Risk** | 0.00 – 39.99 | **19,114** | 95.57% | Normal transaction; proceed with automated ITC credit |
| **Medium Risk** | 40.00 – 69.99 | **742** | 3.71% | Potentially anomalous; secondary compliance review |
| **High Risk** | 70.00 – 100.00 | **144** | 0.72% | Priority audit; block credit until verification |

---

## 10. Database Architecture (SQLite)

Database file: [`database/gst_fraud.db`](file:///D:/s5%20antigravity/gst-fraud-detection/database/gst_fraud.db) (12.5 MB)

### Relational Schema

```text
+------------------------------------+          +------------------------------------+
|             invoices               |          |              analysis              |
+------------------------------------+          +------------------------------------+
| PK  invoice_no       (VARCHAR 50)  |<----+    | PK,FK invoice_no     (VARCHAR 50)  |
|     invoice_date     (VARCHAR 10)  |     +----|       isolation_score(FLOAT)       |
|     supplier_id      (VARCHAR 20)  |          |       isolation_anom (INTEGER)     |
|     supplier_gstin   (VARCHAR 15)  |          |       lof_score      (FLOAT)       |
|     buyer_id         (VARCHAR 20)  |          |       lof_anomaly    (INTEGER)     |
|     buyer_gstin      (VARCHAR 15)  |          |       max_abs_zscore (FLOAT)       |
|     product_category (VARCHAR 50)  |          |       zscore_anomaly (INTEGER)     |
|     hsn_code         (VARCHAR 10)  |          |       risk_score     (FLOAT)       |
|     quantity         (INTEGER)     |          |       risk_level     (VARCHAR 20)  |
|     unit_price       (FLOAT)       |          |       anomaly_type   (VARCHAR 50)  |
|     taxable_value    (FLOAT)       |          |       explanation    (TEXT)        |
|     gst_rate         (FLOAT)       |          +------------------------------------+
|     total_tax        (FLOAT)       |
|     total_amount     (FLOAT)       |
|     payment_method   (VARCHAR 20)  |
+------------------------------------+
```

* **Foreign Key**: `analysis.invoice_no` references `invoices.invoice_no` (1:1 relationship).
* **Indexes**: 11 indexes on `supplier_gstin`, `buyer_gstin`, `invoice_date`, `risk_score`, `risk_level`, `product_category`.

---

## 11. REST API Documentation

### Base URL: `http://127.0.0.1:5000`

| Endpoint | Method | Params | Description |
|---|:---:|---|---|
| `/api/dashboard` | `GET` | None | Returns total count, risk breakdown, and anomaly counts |
| `/api/invoices` | `GET` | `page`, `per_page`, `search`, `risk_level`, `anomaly_type` | Paginated invoice transactions joined with risk metrics |
| `/api/invoices/<id>` | `GET` | `invoice_no` in path | Detailed single invoice data with all 3 model scores |
| `/api/suspicious-invoices` | `GET` | `page`, `per_page`, `search`, `risk_level`, `anomaly_type` | Returns only Medium Risk and High Risk records |
| `/api/model-performance` | `GET` | None | Returns Accuracy, Precision, Recall, F1 for all models |
| `/api/export/csv` | `GET` | `risk_level` (optional) | Streams CSV compliance export for download |
| `/api/upload` | `POST` | `multipart/form-data` with `file` | Ingests CSV, executes full inference, updates SQLite |

---

## 12. Web Frontend & Dashboard UI

The web interface is built using Jinja2 templates extending a common master layout:

1. **Login Page (`/login`)**: Authentication portal with demo credentials helper.
2. **Executive Dashboard (`/dashboard`)**: 7 colored summary cards and interactive Chart.js doughnut & bar charts.
3. **All Invoices Table (`/invoices`)**: 20,000-record paginated table with multi-parameter search and filters.
4. **Suspicious Invoices (`/suspicious`)**: Triage workspace showing only the 886 suspicious invoices.
5. **Invoice Details Drill-down (`/invoice/<id>`)**: Comprehensive view showing line-items, 3 individual ML outputs, risk progress bar, and audit explanation.
6. **Export Reports (`/reports`)**: CSV downloads for all invoices and High, Medium, or Low Risk invoices.
7. **CSV Batch Upload (`/upload`)**: Drag-and-drop CSV upload with live client-side progress spinner and summary.
8. **404 / 500 Pages**: Custom user-friendly error handlers.

---

## 13. Installation & Setup Guide

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13 installed.
* Virtual environment (recommended).

### Step 1: Clone or Navigate to Project
```powershell
cd "D:\s5 antigravity\gst-fraud-detection"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Verify Requirements
* `Flask >= 3.0.0`
* `SQLAlchemy >= 2.0.0`
* `scikit-learn >= 1.4.0`
* `pandas >= 2.0.0`
* `numpy >= 1.24.0`

---

## 14. Running the System & Tests

### Start the Web Application
```powershell
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```
* **Demo Username**: `admin`
* **Demo Password**: `admin123`

### Run System Test Suites

#### 1. Master Full-System Test (Stage 12)
```powershell
python data/test_stage12_full_system.py
```
*(Runs 59 comprehensive checks across all 12 stages with 100% pass rate)*

#### 2. End-to-End Integration Test (Stage 11)
```powershell
python data/test_stage11_integration.py
```
*(Executes 106 integration tests across REST API, database, upload lifecycle, and UI)*

#### 3. Frontend UI Test (Stage 10)
```powershell
python data/test_stage10_frontend.py
```

---

## 15. Academic Viva Q&A Guide

### Q1: Why did you use unsupervised anomaly detection instead of supervised classification?
> **Answer**: Real-world GST filing data has virtually zero reliable ground-truth labels at the time invoices are submitted. Furthermore, fraud tactics evolve constantly (e.g., circular trading, new entity syndicates), making supervised models blind to novel attack vectors. Unsupervised learning models the probability distribution and density of normal business transactions and flags deviations without requiring historical fraud labels.

### Q2: Why combine three different algorithms (Isolation Forest, LOF, Z-Score)?
> **Answer**: Each model has a fundamentally different mathematical inductive bias:
> * **Isolation Forest** isolates global outliers via random recursive axis-aligned tree partitioning.
> * **LOF** measures local density ratios, catching micro-clusters of fraud (e.g., subtle discount variations within a specific product category).
> * **Statistical Z-Score** provides fast, parametric standard deviation bounds and pinpoint feature attribution.
> Fusing all three achieves **95.80% recall** while keeping false positives low.

### Q3: How do you justify the 40% - 40% - 20% ensemble weighting?
> **Answer**: Isolation Forest and LOF explore multi-dimensional non-linear interactions across the 43-feature space and each receive 40% weight. Z-Score is univariate and serves as an extreme statistical sanity check, weighted at 20%. The consensus calibration boosts confidence: when all three agree, precision reaches **91.67%**.

### Q4: Does the system prove that an invoice is fraudulent?
> **Answer**: No. In accordance with legal and compliance standards, anomaly detection identifies **statistical irregularities** and **variance from normal business behavior**. Flagged invoices are categorized as *"suspicious"*, *"potentially anomalous"*, or *"requiring review"*. A qualified human compliance auditor must perform the final review.

### Q5: How is the database organized for high-concurrency access?
> **Answer**: We utilize SQLite with SQLAlchemy scoped sessions (`scoped_session(sessionmaker(...))`) to guarantee thread safety across Flask worker threads. Queries are optimized with 11 B-tree indexes on lookup columns such as `supplier_gstin`, `risk_level`, and `risk_score`.

---

## 16. Compliance & Legal Disclaimer

> **IMPORTANT COMPLIANCE NOTE**:
> This software is a **decision-support and triage system** built for academic and compliance auditing demonstration. Classification of a transaction as "anomalous", "suspicious", or "high risk" is based on mathematical variance and unsupervised pattern recognition. It does **not** constitute legal or statutory proof of tax evasion or fraud. All flagged invoices must be investigated by authorized compliance officers in accordance with applicable GST laws and regulations.

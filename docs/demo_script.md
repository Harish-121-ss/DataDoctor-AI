# DataDoctor AI — 3–5 Minute Demonstration Presentation Script

**Project:** DataDoctor AI  
**Context:** Technical Screening Demonstration (Cognext / CognitionNext '26)  
**Target Audience:** Machine Learning Engineers, Data Architects, Technical Reviewers  
**Target Duration:** 3 to 5 Minutes  

---

## Preparation & Setup
Before presenting:
1. Open PowerShell or Command Prompt in the `DataDoctor-AI` directory and start the server:
   ```bash
   streamlit run app/main.py
   ```
2. Open your web browser at `http://localhost:8501`.
3. Ensure the browser window is maximized (1080p recommended).

---

## Section 1: The Problem & Solution (0:00 – 0:45)

> **Speaker:**
> *"Hello! Every machine learning practitioner has encountered the adage 'Garbage In, Garbage Out.' In real-world enterprise pipelines, dirty tabular data is the number one cause of silent model failure, feature leakage, and corrupted predictions.*
>
> *Most data cleaning either relies on fragile manual scripts, or dangerous automated pipelines that silently drop rows or impute averages without auditability. Moreover, existing tools often conflate two very different things: **syntactic data defects** like missingness or type corruption, versus **statistical anomalies** in high-dimensional feature space.*
>
> *I built **DataDoctor AI** to solve this problem. It is an intelligent tabular data quality diagnosis, anomaly isolation, and safe repair application built with Pandas, Scikit-learn, and Streamlit. It runs 100% locally with zero cloud leakage, enforces strict data immutability, and never applies a transformation without explicit user review and audit logging."*

---

## Section 2: Ingestion & Structural Quality Profiling (0:45 – 1:30)

> **Speaker Action:**
> In the sidebar under **Data Ingestion**, click **"Load Sample Dirty Dataset"** (or drag and drop a dirty CSV). Navigate to **"2. Quality Profiling"**.

> **Speaker:**
> *"Let's see it in action. I'll load our included dirty test dataset.*
>
> *Notice our **Overall Health Score: 78.1 / 100.0**. This score is completely transparent and mathematically calibrated: it starts at 100 and applies specific deductions for missingness, duplicate rows, and defect severity.*
>
> *Looking at our dashboard:*
> - *We have missing cells in the `department` column, clearly highlighted in our Plotly missingness distribution.*
> - *We have 1 exact duplicate row.*
> - *And down in our Defect Catalog, DataDoctor AI has identified 5 concrete defects: including type contamination where currency symbols like `$85,000` disguise numeric fields as text, a constant zero-variance column `status`, and a suspicious negative value in the `age` column.*
>
> *Everything is explained with concrete counts, percentages, and remediation advice."*

---

## Section 3: Multidimensional Anomaly Isolation (1:30 – 2:30)

> **Speaker Action:**
> Navigate to **"3. Anomaly Detection"**. Select features `age` and `salary`. Keep algorithm as **"Isolation Forest (Scikit-learn ML)"** with 5% contamination. Click **"Run Anomaly Detection"**.

> **Speaker:**
> *"Now let's examine anomalous records. This is where DataDoctor AI bridges classical data hygiene and genuine Machine Learning.*
>
> *We configure an unsupervised **Isolation Forest** using Scikit-learn. When we run it on `age` and `salary`, it isolates observations in low-density feature spaces.*
>
> *Take a look at the bivariate scatter plot: the model flagged an extreme record with an unusual age-to-salary ratio.*
>
> *Now, here is a critical architectural principle of DataDoctor AI: **Statistical anomalies are not automatically errors or fraudulent records.** An anomaly merely represents an observation that is statistically rare. In real life, that could be a top-performing executive or a genuine scientific discovery. Therefore, DataDoctor AI never automatically purges anomalies. Instead, we offer non-destructive metadata tagging or user-reviewed filtering."*

---

## Section 4: Safe Reviewable Cleaning & Audit Trail (2:30 – 3:45)

> **Speaker Action:**
> Navigate to **"4. Reviewable Cleaning"**. Click **"Generate Cleaning Proposals"**.
> Expand Proposal `PROP-DEDUP-001` (Deduplication) and click **"Preview Changes"**.
> Click **"Approve & Apply"**.
> Expand Proposal `PROP-CAST-NUM-002` (Currency sanitization) and click **"Approve & Apply"**.

> **Speaker:**
> *"Now we move to remediation. In DataDoctor AI, we follow a strict safety protocol: **Propose -> Preview -> Explicitly Approve -> Audit**.*
>
> *Clicking 'Generate Cleaning Proposals' creates structured proposals directly from our diagnostic and anomaly findings.*
>
> *Before applying deduplication, I click **'Preview Changes'**. This runs a dry-run on an isolated copy of the data. I can see exactly what will happen: 10 rows -> 9 rows, with the exact diff of the removed record. The actual dataset is still untouched.*
>
> *Once satisfied, I click **'Approve & Apply'**. Notice the explicit button action: the engine refuses to execute without explicit user approval.*
>
> *Next, I'll approve our numeric type conversion proposal, which strips stray currency signs and safely casts `salary` to numeric float.*
>
> *Immediately, our **Active Health Score** jumps from **78.1 to 93.8**!*
>
> *Below, our **Structured Audit Trail** records every approved action with an immutable UTC timestamp, row deltas, and affected cells. And our Before-and-After Quality Comparison breaks down the exact trade-offs—such as noting that dropping rows reduces sample power."*

---

## Section 5: Exports, Empirical Benchmark & Conclusion (3:45 – 4:45)

> **Speaker Action:**
> Navigate to **"5. Export & Audit Reports"**. Highlight the download buttons for Cleaned CSV, Original Untouched CSV, Markdown Audit Report, and JSON Audit Log.
> Briefly return to tab **"3. Anomaly Detection"** and expand **"View Reproducible Ground-Truth Benchmark Results"**.

> **Speaker:**
> *"Finally, in Export & Audit Reports, we provide clean, separated exports:*
> 1. *The **Cleaned CSV** for downstream training,*
> 2. *The **Untouched Original CSV** for compliance,*
> 3. *A comprehensive **Markdown Quality Audit Report** with complete mathematical formulas and remediation notes, and*
> 4. *A machine-readable **JSON Audit Log** for enterprise pipeline orchestration.*
>
> *To validate our methods rigorously, we evaluated Isolation Forest against classical Z-Score and Tukey's IQR baselines on a ground-truth synthetic benchmark of 1,000 observations with injected correlation violations:*
> - *Univariate baselines are extremely fast (<35 ms), but miss over 40% of anomalies because individual feature values remain within 1D fences.*
> - *Isolation Forest achieves 78% precision and recall because it isolates multivariate correlation violations, though it has higher latency (~1,000 ms).*
>
> *In summary, DataDoctor AI combines software craftsmanship, machine learning rigor, and responsible data governance into an auditable tool for tabular data preparation.*
>
> *Thank you, and I look forward to your questions!"*

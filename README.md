# AI B2B Supplier Selection Copilot

An AI-assisted procurement decision-support prototype that converts buyer requirements into structured procurement criteria, screens potential suppliers, ranks eligible options, explains trade-offs, and highlights unusual supplier profiles for further review.

**Built with:** Python · Pandas · Streamlit · Scikit-learn · Optional OpenAI API · Google Colab

**Project type:** B2B Procurement · Supplier Analytics · Decision Support · Explainable Ranking

---

## Table of Contents

- [Overview](#overview)
- [Demo Video](#Demo)
- [Business Problem](#business-problem)
- [Project Objectives](#project-objectives)
- [How the System Works](#how-the-system-works)
- [Key Features](#key-features)
- [Supplier Evaluation Criteria](#supplier-evaluation-criteria)
- [Ranking and Decision Logic](#ranking-and-decision-logic)
- [Machine Learning-Based Risk Screening](#machine-learning-based-risk-screening)
- [Technology Stack](#technology-stack)
- [Dataset and Data Design](#dataset-and-data-design)
- [How to Run](#how-to-run)
- [Business KPIs and Evaluation Metrics](#business-kpis-and-evaluation-metrics)
- [Testing and Validation](#testing-and-validation)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Responsible Use](#responsible-use)
- [Author](#author)

---

## Overview

Supplier selection is an important part of B2B procurement. Buyers must balance price, delivery requirements, quality, available stock, warranty, location, and supplier reliability. Choosing only the cheapest supplier may introduce delivery or quality risks, while selecting a supplier with a high rating may not meet the buyer's budget or quantity requirements.

The **AI B2B Supplier Selection Copilot** is a prototype designed to support this decision-making process.

It combines optional large language model (LLM)-assisted requirement extraction with rule-based validation, weighted supplier ranking, explainable near-match analysis, and machine learning-based anomaly screening.

The goal is to make supplier evaluation more structured and transparent while keeping the final purchasing decision with the human buyer.

> **Important:** The prototype uses synthetic supplier data. Its results demonstrate a decision-support workflow and should not be interpreted as verified recommendations about real suppliers.

## Demo

Watch the screen-recording demo of the AI B2B Supplier Selection Copilot:

[▶️ Watch Project Demo on YouTube](https://youtu.be/2Wt8IQl-gHg)

## Business Problem

Traditional supplier evaluation can involve comparing supplier quotations, checking product specifications, validating delivery timelines, reviewing ratings, confirming stock availability, and evaluating commercial trade-offs.

This can create several challenges:

- **Fragmented evaluation:** Relevant supplier information may need to be compared across multiple criteria.
- **Conflicting priorities:** The lowest price may not correspond to the best overall procurement choice.
- **Hard constraints:** A supplier may be unsuitable if it cannot meet the required quantity, budget, delivery deadline, or minimum quality threshold.
- **Limited explainability:** A ranked list is less useful when buyers cannot understand why a supplier was selected or rejected.
- **Supplier profile anomalies:** Unusual patterns in supplier data may deserve additional verification.
- **Manual effort:** Repeated comparisons can be time-consuming when buyers evaluate many alternatives.

This project explores how structured analytics and AI-assisted interfaces can support a more consistent supplier-selection process.

## Project Objectives

1. Convert procurement requirements into structured fields.
2. Validate requirements against available product and supplier data.
3. Separate mandatory eligibility conditions from preference-based ranking.
4. Rank eligible suppliers using buyer-defined priorities.
5. Explain why a supplier is recommended and why near-match alternatives may fail.
6. Flag unusual supplier profiles for additional due diligence.
7. Demonstrate a repeatable, modular prototype through Google Colab and Streamlit.

## How the System Works

The workflow follows these stages:

**Stage 1 — Procurement request**

The buyer specifies the product, variant, quantity, budget, delivery requirement, minimum rating, warranty requirement, location, and relative priorities as applicable.

**Stage 2 — Requirement extraction**

The optional LLM assistant interprets a natural-language request and maps it to structured procurement fields. Ambiguous or missing information should remain unresolved rather than being silently invented.

**Stage 3 — Requirement validation**

The extracted requirements are checked against allowed product variants, price units, locations, and expected data types.

**Stage 4 — Eligibility screening**

Supplier listings are evaluated against mandatory procurement constraints. Suppliers that satisfy the hard constraints can proceed to ranking.

**Stage 5 — Weighted ranking**

Eligible suppliers are ranked according to the buyer's relative priorities for price, delivery, quality, and reliability.

**Stage 6 — Recommendation and explanation**

The application presents the ranked shortlist and supporting information. When no supplier satisfies every hard constraint, the workflow can return near-match alternatives and explain their failed constraints.

**Stage 7 — Risk screening**

An Isolation Forest-based model identifies supplier profiles that appear unusual relative to the available dataset. These flags are intended to prompt further review, not to establish that a supplier is fraudulent or unreliable.

## Key Features

### 1. Natural-language procurement intake

An optional LLM integration helps transform a buyer's free-text request into structured procurement requirements.

Examples of requirement fields include:

- Product variant
- Required quantity
- Maximum unit price
- Maximum delivery time
- Minimum supplier rating
- Minimum warranty duration
- Location
- Relative importance of price, delivery, quality, and reliability

The LLM is used for requirement interpretation, not as the final supplier-ranking engine.

### 2. Structured supplier comparison

The system evaluates supplier listings using relevant commercial and operational attributes, including:

- Price
- Available stock
- Delivery time
- Supplier rating
- Warranty
- Location
- Overall ranking score

### 3. Hard-constraint screening

Mandatory buyer requirements are treated separately from preference-based ranking. A supplier that fails a mandatory condition should not be presented as a fully eligible match.

### 4. Preference-based ranking

Buyers can prioritize price, delivery, quality, and reliability differently. This allows the ranking to reflect the buyer's procurement objectives rather than applying one fixed preference to every request.

### 5. Explainable recommendations

The project provides information intended to help users understand supplier ranking and selection trade-offs.

Where a perfect match is unavailable, the near-match workflow identifies failed constraints so buyers can understand what would need to change before a supplier could qualify.

### 6. Machine learning-based anomaly screening

An Isolation Forest model screens supplier profiles for unusual patterns relative to the available dataset.

The model generates a relative anomaly score and a screening category. These outputs are indicators for investigation, not calibrated probabilities of supplier failure.

### 7. Interactive Streamlit interface

The project includes a Streamlit application with procurement intake, supplier results, ranking information, and risk-screening outputs.

The notebook generates the application files and launches the interface within Google Colab.

---

## Supplier Evaluation Criteria

| Criterion | Purpose |
|---|---|
| Price | Checks commercial fit against the buyer's budget and price preferences. |
| Available stock | Helps determine whether the required quantity can be supplied. |
| Delivery time | Evaluates whether the supplier can meet the requested timeline. |
| Rating | Represents the supplier rating available in the dataset. |
| Warranty | Helps compare the warranty offered against the buyer's requirement. |
| Location | Supports location-related requirements and delivery evaluation where implemented. |
| Reliability-related attributes | Contribute to the configured supplier evaluation and anomaly-screening process. |
| Overall score | Supports preference-based ranking among eligible suppliers. |

The exact effect of each criterion depends on the validation rules, scoring implementation, available data, and buyer-defined priorities.

## Ranking and Decision Logic

The system separates two different decisions.

### A. Eligibility

Hard constraints determine whether a supplier qualifies as a match.

Examples include:

- Correct product variant
- Sufficient available quantity
- Price within the maximum budget
- Delivery within the required time
- Rating at or above the minimum threshold
- Warranty meeting the stated requirement

The applicable constraints depend on the buyer's request and the rules implemented in the application.

### B. Ranking

Eligible suppliers are ordered according to the buyer's relative priorities.

For example, a buyer purchasing a time-sensitive component may place more importance on delivery reliability than on a small difference in price. Another buyer with a flexible timeline may emphasize price.

The project uses a deterministic supplier-ranking process. The LLM does not independently decide which supplier wins.

**Interpretation note:** The overall score is a relative ranking score. It should not be described as a percentage, a probability of success, or a guarantee of supplier performance.

## Machine Learning-Based Risk Screening

The prototype includes an Isolation Forest-based anomaly-screening component.

### What it does

- Builds supplier-level features from available supplier and listing data.
- Applies feature scaling and Isolation Forest.
- Produces a relative anomaly score on a 0–100 scale.
- Groups profiles into screening categories such as Typical, Elevated, or High, according to the implemented thresholds.
- Provides notes to encourage verification of unusual profiles.

### What the score means

A higher relative anomaly score indicates that a supplier profile appears more unusual relative to the supplier profiles used for comparison.

It does **not** mean that the supplier has the same percentage probability of default, fraud, late delivery, or poor quality.

The model needs sufficient supplier profiles to perform meaningful screening. The current implementation uses a minimum-data check and returns an insufficient-data status when the dataset is too small.

Because the prototype uses synthetic data, its anomaly patterns are useful for demonstrating the method but do not establish real-world predictive accuracy.

---

## Technology Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and supplier comparison |
| NumPy | Numerical processing |
| Streamlit | Interactive user interface |
| Scikit-learn | Feature scaling and Isolation Forest anomaly screening |
| OpenAI API (optional) | Natural-language procurement requirement extraction |
| Google Colab | Notebook execution and prototype environment |
| Jupyter Notebook | Reproducible workflow and code organization |

The notebook installs and generates the required application modules during execution. The exact installed package versions may change unless they are explicitly pinned.

## Dataset and Data Design

The current prototype generates and uses a **synthetic procurement dataset** rather than a verified database of real suppliers.

The notebook creates structured data files representing product information, supplier profiles, and supplier listings.

These datasets support demonstrations of:

- Product and variant matching
- Supplier eligibility checks
- Price and delivery comparisons
- Stock and warranty checks
- Preference-based ranking
- Near-match explanations
- Supplier profile anomaly screening

Synthetic data makes the prototype reproducible and avoids presenting invented supplier records as genuine market information.

**Data limitation:** The results depend on the assumptions and distributions used to generate the synthetic data. They should not be treated as evidence of actual supplier availability, market prices, or supplier performance.

## How to Run

### Option 1 — Run through Google Colab

1. Open the notebook in this repository: `AI_B2B_Supplier_Selection_Copilot_Final.ipynb`.
2. Download the notebook or open it directly in Google Colab.
3. Select a Colab runtime.
4. Run the notebook cells from top to bottom in the intended order.
5. Allow the setup cells to install dependencies and generate the required application files and synthetic data.
6. If using natural-language requirement extraction, provide a valid OpenAI API key through the app's password field or the supported environment-variable configuration.
7. Launch the Streamlit application using the notebook's launch cell.
8. Open the served application interface in Colab.
9. Enter procurement requirements and review the resulting supplier recommendations and explanations.

The exact execution time depends on dependency installation, runtime availability, and whether the optional LLM service is used.

### OpenAI API configuration

The natural-language assistant requires an API key when that feature is used.

The application supports entering a key through its password-style input or configuring the `OPENAI_API_KEY` environment variable.

Do not commit API keys to GitHub or place them directly in notebook code. API usage may incur charges according to the provider's current pricing and account settings.

Supplier ranking is deterministic and is separate from the optional LLM requirement-extraction step.

### Option 2 — Local execution

The notebook currently serves as the setup and launch workflow for the application. A clean local installation may require extracting the generated application modules and data files into a project directory, installing their dependencies, configuring optional credentials, and launching Streamlit.

A fully documented local setup should be added once the generated modules, data files, and dependency versions are packaged and tested independently of Colab.

## Business KPIs and Evaluation Metrics

The current project exposes supplier-level evaluation measures and dataset-status metrics. These are useful for supplier comparison, but they are not the same as measured procurement business outcomes.

### Metrics used in supplier evaluation

| Metric | How it is used |
|---|---|
| Unit price | Compare commercial offers and enforce budget limits. |
| Delivery days | Compare delivery time and enforce deadline requirements. |
| Rating | Compare the rating available in the dataset. |
| Warranty | Compare warranty duration with buyer requirements. |
| Available stock | Check quantity feasibility. |
| Failed constraints | Explain why a near-match supplier does not qualify fully. |
| Overall score | Rank eligible suppliers according to configured priorities. |
| Relative anomaly score | Highlight unusual supplier profiles for additional review. |

The application also displays dataset-status counts, such as suppliers, supplier listings, and product variants.

### Business KPIs to add in a future version

These are proposed enhancements and should not be described as already implemented unless the application is updated to calculate them.

- **Potential cost savings:** Compare a selected supplier's cost with a clearly defined baseline quotation or benchmark.
- **On-time delivery rate:** Percentage of actual orders delivered by their agreed delivery dates.
- **Defect rate:** Percentage of received units that fail an agreed quality standard.
- **Supplier rejection rate:** Percentage of supplier offers rejected against mandatory criteria.
- **Average selection time:** Time required to complete a supplier-selection task.
- **Supplier concentration:** Share of procurement spend or order volume concentrated among suppliers.
- **Risk review rate:** Share of flagged supplier profiles sent for additional due diligence.

These KPIs require appropriate baselines, order history, delivery outcomes, quality data, or recorded user activity. Synthetic supplier listings alone are not sufficient to establish actual savings or operational improvement.

## Testing and Validation

The notebook includes smoke-test logic for the supplier-selection workflow, including:

- Running a standard procurement request.
- Checking the resulting status and ranked or near-match records.
- Testing a strict requirement set where a perfect match is unavailable.
- Checking that an explanatory result is returned for the no-perfect-match path.
- Checking integration of the ML anomaly-screening output.

Tests should be rerun after code changes and in a fresh runtime before presenting a new release.

Passing prototype tests demonstrates that the tested workflow behaves as expected for the tested inputs. It does not establish the predictive accuracy of the anomaly model or validate performance with real procurement data.

## Limitations

- Supplier and listing data are synthetic, not verified live supplier data.
- The prototype does not independently establish real-world supplier legitimacy.
- The ranking is dependent on configured criteria, weights, and data quality.
- The LLM can misinterpret ambiguous buyer requests; structured validation and user review remain important.
- The anomaly model detects unusual patterns rather than confirmed misconduct or future supplier failure.
- Relative anomaly scores are not probabilities of default or performance.
- No actual cost savings, on-time delivery improvement, or reduction in procurement time has been demonstrated without outcome data.
- The notebook is designed around a Google Colab workflow; standalone deployment needs separate packaging and validation.
- The application should not replace commercial due diligence, contract review, quality audits, or human purchasing approval.

## Future Enhancements

Potential future improvements include:

1. **Procurement KPI dashboard:** Track cost, delivery, quality, and sourcing outcomes when reliable historical data is available.
2. **Historical supplier performance:** Incorporate actual order, delivery, defect, and rejection records.
3. **Scenario analysis:** Compare how the shortlist changes when price, delivery, quality, or reliability priorities change.
4. **Sensitivity analysis:** Show whether small changes in buyer priorities materially change the top-ranked supplier.
5. **Supplier comparison reports:** Export a structured report with rankings, failed constraints, and reasons for selection.
6. **Real supplier-data integration:** Connect to verified, authorized supplier data sources and implement data-quality checks.
7. **Persistent data storage:** Replace temporary runtime files with a controlled storage layer.
8. **Reproducible deployment:** Add a dependency file, organized source modules, configuration instructions, and a local launch guide.
9. **Model evaluation:** Assess anomaly screening with labeled, representative data before making predictive claims.
10. **Audit trail:** Record requirement changes, ranking weights, recommendations, and human overrides.

## Responsible Use

This tool is intended to support procurement decisions, not make autonomous purchasing decisions.

Users should validate supplier information, confirm stock and delivery commitments, review contractual terms, and investigate unusual profiles before onboarding or placing orders.

Supplier rankings should be interpreted in the context of the buyer's requirements and the quality of the underlying data.

## Author

**Project:** AI B2B Supplier Selection Copilot

**Repository:** [AI_B2B_Supplier_Selection_Copilot](https://github.com/shivamsahu-7/AI_B2B_Supplier_Selection_Copilot)

A portfolio project exploring the application of AI-assisted requirement extraction, deterministic decision logic, supplier analytics, and anomaly screening to B2B procurement.

---

*This README describes a working prototype and distinguishes implemented functionality from proposed enhancements. Update it as the codebase and deployment approach evolve.*

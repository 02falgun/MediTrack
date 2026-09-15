# MediTrack Documentation Hub

Welcome to the documentation for **MediTrack — Clinical Decision Support & 30-Day Readmission Risk System**.

### Primary Guides & Manuals
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**: **Start here!** The master guide explaining the entire project in plain English, real-world healthcare context, step-by-step technical workings, tech stack, database schema, machine learning models, explainable AI (SHAP), clinical safety guardrails, web dashboard, and FAQs.

### Core System Artifacts
- **Relational Database**: [schema.sql](../sql/schema.sql) and [queries.sql](../sql/queries.sql)
- **Clinical Knowledge Base & Safety**: [src/clinical_rag.py](../src/clinical_rag.py)
- **Machine Learning & Preprocessing**: [src/model_pipeline.py](../src/model_pipeline.py)
- **FastAPI Backend Service**: [app/main.py](../app/main.py)
- **Clinical Dashboard Frontend**: [app/static/index.html](../app/static/index.html) and [app/static/app.js](../app/static/app.js)
- **Audits & Analytical Extensions**: [src/extensions.py](../src/extensions.py)
- **Executive Slide Deck**: [slides/meditrack_presentation_slides.pdf](../slides/meditrack_presentation_slides.pdf)
- **Clinical Dashboard Quality Report**: [reports/meditrack_clinical_dashboard.pdf](../reports/meditrack_clinical_dashboard.pdf)

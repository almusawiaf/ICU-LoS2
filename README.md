# Improving Hospital Length of Stay Prediction through Heterogeneous Data Integration from MIMIC-III Records

## Introduction

This repository presents a comprehensive pipeline for processing structured and unstructured clinical data extracted from the MIMIC-III database and applying advanced machine learning techniques for the prediction of Intensive Care Unit (ICU) Length of Stay (LoS). The project leverages domain-specific embeddings, summarization methods, and graph-based learning in a high-performance computing (HPC) environment.

## Problem Statement

Accurate prediction of ICU length of stay is critical for improving resource allocation, patient care planning, and clinical decision-making. While structured data (e.g., diagnoses, procedures, labs) provide important signals, clinical notes contain rich, latent information that remains underutilized. This project addresses the challenge of integrating structured and unstructured data for LoS prediction using advanced embedding and summarization strategies, and deploying scalable machine learning workflows on HPC systems.

## Repository Structure

This repository is divided into two primary components:

---

### 1. `Data_Preparation/`

This directory contains all scripts and SLURM job files necessary for the extraction, processing, and unification of clinical features.

**Subcomponents:**

- `1.1 Extracting Structured Features, (ZSF)`: Scripts to extract, normalize, and encode the structured features. Structured features—such as diagnoses, procedures, medications, labs, microbiology, vitals, and demographics—were extracted and preprocessed individually. Feature selection (e.g., ANOVA F-test) was applied per modality, and selected features were concatenated into a unified representation. Continuous features were z-score normalized. The final composite vector, referred to as ZSF, captures clinically relevant structured signals for downstream modeling.
  
- `1.2 Extracting Unstructured Features, (E)`: 
  - Preprocessing of clinical notes from the MIMIC-III NOTEEVENTS table.
  - Symptom extraction using MeSH ontology.
  
- `1.2.1 Summarization Techniques`:
  - Includes methods for summarizing long clinical notes using models such as:
    - `T5-small`
    - `LongChat`
    - `BART-large-CNN`
    - `Medical-Summarization` (domain-specific)

- `1.3 Clinical Embedding Extraction`:
  - Generation of text embeddings using:
    - `Bio_ClinicalBERT`
    - `ClinicalBERT`
    - Other biomedical transformer-based models.

- `1.4 Unified Feature Representation`:
  - Integration of structured and unstructured features into a consolidated format suitable for modeling.
  
- `1.5 Lung Cancer Subset Extraction`:
  - Pipeline for extracting a targeted lung cancer cohort for specialized experiments or external validation.

---

### 2. `Machine_Learning_Training/`

This directory includes model training configurations and job scripts optimized for execution on high-performance clusters.

**Subcomponents:**

- `2.1 Binary Classification`:
  - Code and scripts to train models for binary LoS classification (e.g., short vs. long stay).

- `2.2 Multi-class Classification`:
  - Training pipelines for three-class LoS prediction (e.g., short, medium, long stay).

Each training mode supports classical models (e.g., ANN, XGBoost) and graph-based architectures (e.g., GCN, GAT, GraphSAGE), with compatibility for SLURM-based job scheduling and distributed computation.

---

## Execution Notes

- All major tasks are modularized as SLURM job files (`.sh`) to support batch execution in HPC environments.
- The scripts assume environment modules and conda environments for PyTorch, HuggingFace Transformers, and PyTorch Geometric.

---

## Citation

To be shared soon.
---

## Contact

For inquiries or collaborations, contact Ahmad F. Al Musawi at [almusawiaf@utq.edu.iq](mailto:almusawiaf@utq.edu.iq) or [almusawiaf@vcu.edu](mailto:almusawiaf@vcu.edu).

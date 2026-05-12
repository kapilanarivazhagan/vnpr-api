# Vehicle Number Plate Recognition & Verification API

A FastAPI-based ANPR (Automatic Number Plate Recognition) system designed for vehicle identity verification, OCR-based plate recognition, and operational validation workflows.

---

## Overview

This project provides an API for detecting and recognizing vehicle number plates from uploaded images while validating recognized plates against assigned vehicle identities.

The system is designed for operational verification workflows, fraud prevention, and vehicle identity validation use cases.

---

## Key Features

- Vehicle number plate recognition
- OCR-based plate extraction
- Vehicle identity verification
- Similarity score validation
- Confidence-level classification
- FastAPI backend architecture
- Secure API authentication
- Operational verification workflows

---

## Tech Stack

### Backend & APIs
- FastAPI
- Python
- Uvicorn

### Computer Vision & OCR
- OpenCV
- OCR Pipelines
- NumPy

### Verification & Processing
- Similarity Matching
- Confidence Scoring
- Validation Logic

---

## System Workflow

```text
Vehicle Image
      ↓
Plate Detection
      ↓
OCR Extraction
      ↓
Vehicle Number Recognition
      ↓
Similarity Validation
      ↓
Verification Decision

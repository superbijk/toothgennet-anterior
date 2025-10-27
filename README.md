# A Latent Variable Deep Generative Model for 3D Anterior Tooth Shape

**Authors:** Chawalit Chanintonsongkhla, Varin Chouvatut, Chumphol Bunkhumpornpat, Pornpat Theerasopon  
**Journal:** *Journal of Prosthodontics*, 2025  
**DOI:** [10.1111/jopr.14092](https://doi.org/10.1111/jopr.14092)  
**PubMed:** [PMID 40624318](https://pubmed.ncbi.nlm.nih.gov/40624318/)

---

## Overview

This repository accompanies the research paper  
**"A Latent Variable Deep Generative Model for 3D Anterior Tooth Shape"**.

The study presents a latent variable deep generative approach—**PointFlow**—for synthesizing realistic 3D anterior tooth geometries. The model learns morphological features from natural tooth scans and reconstructs clinically accurate shapes suitable for digital prosthodontic workflows.

*The complete dataset, trained models, and original implementation will be released following publication.*

---

## Abstract

**Purpose:**  
To introduce a 3D generative model capable of producing realistic anterior tooth shapes and to assess its potential for clinical reconstruction.

**Materials and Methods:**  
A dataset of 1337 3D scans of natural anterior teeth was used to train a deep generative model, PointFlow. The model encodes complex geometries into a compact latent space, enabling generation and reconstruction of new shapes. Performance was evaluated using seven 3D shape metrics, and the model’s applicability was tested by reconstructing 60 artificially damaged samples.

**Results:**  
PointFlow successfully modeled the diversity of anterior tooth shapes, outperforming baseline datasets across multiple generative metrics. Reconstruction tasks achieved an average Chamfer Distance of 0.2738 ± 0.095 mm for missing regions.

**Conclusions:**  
Deep generative modeling effectively learns natural tooth morphology and enables the synthesis of high-quality 3D geometries suitable for clinical use.

---

## Data and Code Availability

- **Training Dataset:** 1337 aligned 3D anterior tooth scans  
- **Reconstruction Dataset:** 60 artificially damaged samples  
- **Implementation:** Full PointFlow training and inference code  
- **Release Schedule:** Complete dataset and original implementation will be made publicly available following publication (expected Q1 2026)

---

## Citation

If you use this work, please cite:

```bibtex
@article{Chanintonsongkhla2025PointFlow,
  title     = {A Latent Variable Deep Generative Model for 3D Anterior Tooth Shape},
  author    = {Chanintonsongkhla, Chawalit and Chouvatut, Varin and Bunkhumpornpat, Chumphol and Theerasopon, Pornpat},
  journal   = {Journal of Prosthodontics},
  year      = {2025},
  doi       = {10.1111/jopr.14092}
}

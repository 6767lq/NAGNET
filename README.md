# NAGNet: Measurement-Consistent Low-Field MRI Enhancement via k-Space Noise Modeling

<p align="center">
  <img src="fig/fig2_01.png" alt="NAGNet Overview" width="800">
</p>

This repository contains the official PyTorch implementation of the paper:

> **Measurement-Consistent Low-Field MRI Enhancement via k-Space Noise Modeling**  
> *Yutao Hu, Qi Liu, Tao Zhou, Ziru Li, Ping Liang, Jichang Zhang, Rongsheng Lu, Xiaojian Lou, Jianjun Zheng, Jian Yang, Yang Chen*  


---

## 📌 Abstract

Low-field MRI has gained increasing attention as a cost-effective and portable alternative to high-field MRI. However, its inherently low magnetic field strength leads to reduced signal intensity and low SNR, resulting in severe image degradation. This work presents a **measurement‑consistent framework** that addresses three key aspects:

- **Data Synthesis** – physics‑inspired k‑space noise modeling with independent real/imaginary Gaussian injection.
- **Model Design** – a Noise‑Aware Guidance Network (NAGNet) with explicit spatially‑varying noise supervision.
- **Evaluation** – a unified benchmark including real paired acquisitions from 0.1 T, 0.35 T and 1.5 T scanners.

**Key contributions:**

- ✅ Physics‑based k‑space synthesis that preserves Fourier phase coherence.
- ✅ **Open‑release of real low‑field MRI datasets** (0.1 T, 0.35 T) with corresponding 1.5 T references from the same subjects.
- ✅ NAGNet with an Adaptive Noise Awareness (ANA) branch that explicitly estimates noise maps.
- ✅ State‑of‑the‑art performance across synthetic and real benchmarks.
- ✅ Strong generalisation to unseen field strengths (0.05 T, 0.1 T, 0.35 T) and different anatomical regions.
- 
## 📂 Data Availability – **Open for Research**

**This is a core contribution of our work.** We are releasing the real‑world datasets that enable fair and reproducible benchmarking in low‑field MRI enhancement:

- **PriReL‑0.1 / PriReL‑0.35** – Real clinical brain scans acquired at 0.1 T and 0.35 T, together with **high‑field (1.5 T) references** from the same subjects (anatomically aligned but not perfectly registered). This is the **first publicly available** cross‑field paired dataset of its kind.


> **Data download** – The curated datasets will be uploaded to a public repository (link to be provided upon paper acceptance). Usage is restricted to **non‑commercial research purposes**; a detailed data use agreement will accompany the release.

## 🚀 Key Features

### 1. Measurement‑Consistent k‑Space Synthesis

Unlike image‑domain noise addition, our pipeline:

- Estimates noise variance from the **high‑frequency annular region** of real low‑field k‑space.
- Injects **independent Gaussian noise** into real and imaginary components.
- Preserves the complex‑valued signal statistics, maintaining phase consistency.

### 2. NAGNet Architecture
NAGNet consists of two synergistic paths:

Restoration backbone – reconstructs fine anatomical details.

Adaptive Noise Awareness (ANA) branch – a lightweight encoder‑decoder that predicts a spatially‑varying noise map.
The ANA branch is explicitly supervised by the known noise from k‑space synthesis, giving its output a clear physical meaning. It can be plugged into any restoration backbone to consistently improve denoising performance.

# Transient Signal Propagation in Heterogeneous Networks under Various Inputs

**Authors:** Xiaoge Bao, Wei P. Dai, Jan Nagler, and Wei Lin

---

## Description

This repository contains the code used to produce the main results in the paper *Transient Signal Propagation in Heterogeneous Networks under Various Inputs*. The code is organized **following the order of the main figures** in the article and does **not** cover all the appendix or supplementary materials.

The central purpose of the code is to compare **theoretical estimations with simulated results**. Section II specifically examines the accuracy of the theoretical predictions. For readers interested in the practical linkage to brain networks, the folders for **Figs. 7 and 8** contain tutorial-style code that demonstrates how the framework can be applied to understand network structure in propagation for linear neural dynamics.

---

## Code Organization

Each main figure corresponds to a dedicated folder or script:

| Figure(s) | Location | Content |
|-----------|----------|---------|
| Figs. 2 & 5 | `Fig2&5_GeneralMetrics/` | Comparison of theoretical predictions and simulations for general network metrics under diverse topologies |
| Fig. 4 | `Fig4_PrimateCortex/` | Application in real-SC-constrained primate cortical model |
| Fig. 6 | `Fig6_Truncation/` | Truncation analysis in networks with homogeneous intrinsic time constant |
| Fig. 7 | `Fig7_FFvsFB/` | Comparison between feedforward and feedback networks |
| Figs. 8 & 9 | `Fig8&9_WilsonCowan/` | Tutorial-style code for understanding structural effects in linear neural dynamics |

---

## Getting Started

**Requirements.** Python ≥ 3.8. Install dependencies:

```bash
pip install -r requirements.txt

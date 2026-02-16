# Rethinking Virtual Screening From the Lenses of Imbalanced Regression

Virtual screening increasingly relies on continuous potency prediction to prioritize compounds for experimental testing, yet molecular activity data are highly imbalanced, with only a small fraction of compounds exhibiting chemically relevant high potency. Standard regression models and evaluation metrics optimize global accuracy and often fail to capture performance in these critical regions. In this work, we systematically study the impact of imbalance-aware regression strategies on virtual screening across ten activity prediction tasks from ChEMBL. We evaluate k-Nearest Neighbors, Support Vector Regression, Extreme Gradient Boosting, and Deep Neural Networks trained with both standard and imbalance-aware objectives, including Label Distribution Smoothing, RankSim, Balanced Mean Squared Error, Squared Error–Relevance Area, and Autofocused Oracle. Performance is assessed using regression accuracy, threshold-aware classification, ranking fidelity, and early enrichment metrics. Our results show that imbalance-aware methods consistently improve identification, ranking, and enrichment of highly potent compounds, highlighting their importance for effective virtual screening with continuous potency prediction.

---

## Repository Structure (expected)
- `ml_models.py` — main script to generate results
- `requirements.txt` — Python dependencies
- `dataset/chembl_30_IC50_10_tids_1000_CPDs.csv` — dataset file (make sure it exists in this path)

---

## Requirements

### System / GPU
- CUDA: **11.6**
- cuDNN: **8.0.4**

### Python dependencies
Install dependencies with:
```bash
pip install -r requirements.txt

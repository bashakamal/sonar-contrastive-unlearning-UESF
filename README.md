# Supervised Contrastive Machine Unlearning of Background Bias in Sonar Image Classification with Fine-Grained Explainable AI

Official PyTorch implementation of our CVIP 2025 paper:

> **Supervised Contrastive Machine Unlearning of Background Bias in Sonar Image Classification with Fine-Grained Explainable AI**  
> Kamal Basha S, Athira Nambiar  
> SRM Institute of Science and Technology

This repository implements:

- **Baseline sonar classifier** using EfficientNet-B0
- **Targeted Contrastive Unlearning (TCU)** with triplet loss (anchor, positive, *seafloor* negative)
- **Unlearn-to-Explain Sonar Framework (UESF)** using LIME-based differential explanations  
- Evaluation on **real + S3Simulator+ synthetic** sonar data (plane, ship, mine, human, seafloor)

---

## 1. Installation

```bash
git clone https://github.com/<your-username>/sonar-contrastive-unlearning-UESF.git
cd sonar-contrastive-unlearning-UESF

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

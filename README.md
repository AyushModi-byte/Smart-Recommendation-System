# Smart-Recommendation-System

# AuraPath 🗺️
### Multi-Modal Semantic Trajectory Synthesis for Aesthetic-Driven POI Recommendation

> *Beyond "nearest coffee shop" — AuraPath recommends places that match your visual soul.*

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🧠 What is AuraPath?

Traditional navigation systems recommend the **closest** or **most popular** locations. AuraPath goes deeper — it analyzes your **visual history** to extract your *Visual DNA*, combines it with your **psychological mood** and **real-world context**, and recommends highly specific aesthetic-driven locations like:

> *"Minimalist Industrial Coffee Shop"* instead of just *"Coffee Shop"*

---

## 🏗️ Architecture

AuraPath is built on a **3-module pipeline**:

### Module A — The Eyes (Multi-Level Visual Encoder)
- Pre-trained **ResNet50** extracts 2048-dim visual embeddings
- **MEAL Framework** + **Quintuplet Loss** fine-tunes embeddings in semantic space
- **Self-Attention** mechanism filters high-quality place photos from junk

### Module B — The Brain (Contextual Clustering Engine)
- **ICA-FCM Hybrid** (Imperialist Competitive Algorithm + Fuzzy C-Means)
- Dynamically discovers optimal K=6 aesthetic clusters
- Assigns **fuzzy probabilities** to each POI (e.g. 60% Historic, 40% Relaxing)

### Module C — The Decision Maker (Fusion & Prediction)
- Concatenates visual + contextual embeddings into unified user/POI representations
- **Neural Interaction Layer** with element-wise product captures cross-feature interactions
- **Sigmoid** output predicts final visit probability for unexplored locations

---

## 📊 Results

| Metric | Score |
|---|---|
| AUC-ROC | **0.9978** |
| Hit Rate@20 | **37.6%** |
| NDCG@10 | **0.185** |
| Recall@10 | **21.1%** |
| Precision@10 | **5.5%** |
| Val Accuracy | **85.8%** |

---

## 🗂️ Dataset

- **YFCC15M** (Yahoo Flickr Creative Commons 100M subset)
- Filtered for geo-tagged media in **USA & Europe**
- ~3.7M photos across **201,623 POIs** from **27,071 users**
- After quality filtering: **767,210 photos** across **169,845 POIs**

---

## 🎨 Discovered Aesthetic Clusters

| Cluster | Label | POIs |
|---|---|---|
| 0 | Live Events & Recreation | 21,983 |
| 1 | Urban & Street Architecture | 29,928 |
| 2 | Historic & Rural Heritage | 22,883 |
| 3 | Scenic Nature & Waterscapes | 28,920 |
| 4 | Nightlife & Social Scenes | 33,361 |
| 5 | Green Nature & Wildlife | 32,770 |

---

## 📓 Notebooks

| # | Notebook | Description |
|---|---|---|
| 1 | NB1_Data_Acquisition | Download & setup YFCC15M dataset |
| 2 | NB2_Cleaning_Filtering | Clean & filter geo-tagged USA/Europe data |
| 3 | NB3_ResNet50_Embeddings | Extract 2048-dim visual embeddings |
| 4 | NB4_Contextual_Embeddings | Generate 391-dim contextual embeddings |
| 5 | NB5_DBSCAN_POI_Mapping | Assign POI IDs via DBSCAN clustering |
| 6 | NB6_MEAL_Quintuplet_Loss | Fine-tune embeddings with Quintuplet Loss |
| 7 | NB7_SelfAttention_Filtering | Filter quality place photos via Self-Attention |
| 8 | NB8_ICA_FCM_Clustering | ICA-FCM hybrid clustering for aesthetic groups |
| 9 | NB9_Fuzzy_POI_Labeling | Assign aesthetic labels to clusters |
| 10 | NB10_Neural_Interaction | Train Neural Interaction Layer |
| 11 | NB11_Evaluation | Full evaluation with Recall, NDCG, AUC-ROC |
| 12 | NB12_Inference_Pipeline | End-to-end demo — recommend by user or mood |

---

## ⚙️ Setup

```bash
# Clone repo
git clone https://github.com/yourusername/AuraPath.git
cd AuraPath

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
# Recommend by User ID
demo_user("user_id_here", top_k=5)

# Recommend by Mood/Aesthetic
demo_mood(3, top_k=5)  # 3 = Scenic Nature & Waterscapes
```

---

## 🖥️ Hardware

| Node | Specs | Role |
|---|---|---|
| Local | Lenovo IdeaPad, AMD Ryzen 7 4800H, 8GB RAM | Data management |
| DGX Server | 8× Tesla V100 32GB, multi-core CPU | Heavy compute |

---

## 👤 Author

**23DCS510** — B.Tech Computer Science  
*Smart Recommendation System Project*

---

## 📄 Citation

If you use this work, please cite:
```
@project{aurapath2026,
  title   = {AuraPath: Multi-Modal Semantic Trajectory Synthesis for Aesthetic-Driven POI Recommendation},
  authors  = {Ayush Modi , Anurag Yadav},
  year    = {2026}
}
```

# CLIPSitu-Vid: Extending CLIPSitu for Video-Based Situation Recognition

This repository extends [CLIPSitu](https://github.com/LUNAProject22/CLIPSitu) to the **VidSitu** dataset, enabling **video-based situation recognition**. CLIPSitu-Vid builds upon **CLIPSitu’s CLIP-based semantic role labeling** and integrates the **VidSitu dataset**, making it the first adaptation of CLIPSitu for videos.

Additionally, we **compare model performance with a Vision-Language Model (VLM)** using **VILA** via zero-shot prompting in notebooks, but **VILA is not integrated into the main model**.

## 🚀 Overview

**CLIPSitu-Vid** is designed to understand **multi-frame situation recognition**. Given a **video clip**, the model predicts **verbs, roles, and objects** at multiple timestamps, following the **VidSitu** annotation framework.

### 🔑 Key Features

- **Video Situation Recognition**: Predicts **structured event representations** (verbs, roles, objects) for **each 2-second segment** in a video.
- **Transformer with Temporal Encoding**: Extends CLIPSitu’s **image-based CLIP embeddings** with **video-aware multi-frame reasoning**.
- **VLM Comparison**: Uses **VILA for zero-shot evaluations** in notebooks to compare performance against our approach.
- **Structured Outputs**: Generates structured JSON outputs, compatible with knowledge-based reasoning systems.
- **Efficient Data Handling**: Large datasets are hosted externally on **Google Drive** for easier access.

---

## 📚 Dataset

### **VidSitu**
- A large-scale video dataset (**29,000 movie clips**) annotated at **2-second intervals** with **verbs, roles, and entities**.
- Each video contains **five events**, with **semantic role labels** for each verb.
- For more details, visit the [VidSitu repo](https://github.com/TheShadow29/VidSitu).

### **GVSR Framework**
- We build upon **GVSR**, a graph-based model for **video-based semantic role labeling**.
- GVSR enhances **verb and role prediction** with a graph-based temporal structure.
- More details available in the [GVSR repo](https://github.com/zeeshank95/GVSR).

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/CLIPSitu-Vid.git
cd CLIPSitu-Vid
pip install -r requirements.txt

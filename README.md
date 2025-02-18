# CLIPSitu-Vid: Extending CLIPSitu for Video-Based Situation Recognition

This repository extends [CLIPSitu](https://github.com/LUNAProject22/CLIPSitu) to the **VidSitu** dataset, enabling **video-based situation recognition**. CLIPSitu-Vid builds upon **CLIPSitu’s CLIP-based semantic role labeling**, integrates the **VidSitu dataset**, and evaluates **Vision-Language Models (VLMs)** like **VILA** via zero-shot prompting.

This work is based on our **journal paper**:  
📄 **[Effectively Leveraging CLIP for Generating Situational Summaries of Images and Videos](https://arxiv.org/pdf/2407.20642)** (2024).

---

## 🚀 Overview

**CLIPSitu-Vid** is designed to perform **multi-frame situation recognition**, where it predicts **verbs, roles, and objects** at multiple timestamps within a video.

### 🔑 Key Features

- **Video Situation Recognition**: Predicts **structured event representations** (verbs, roles, objects) for **each 2-second segment** in a video.
- **Transformer with Temporal Encoding**: Extends CLIPSitu’s **image-based CLIP embeddings** with **video-aware multi-frame reasoning**.
- **VLM Evaluation for Benchmarking**: Compares **zero-shot prompting from VILA** against our structured approach.
- **Structured Outputs**: Generates JSON-based outputs, compatible with downstream reasoning systems.

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

## 📂 Handling Large Files

### **Download Dataset and Models**
Large datasets and models are stored on **Google Drive** to keep the repository lightweight.  
Download the required files from the link below:

📂 **[Google Drive: CLIPSitu-Vid Datasets & Models](https://drive.google.com/drive/folders/1mUqBRu6-ncGz65LHAaEeGP6tQox-tyGI)**  

## 📊 Model Architecture

### **CLIP-Based Transformer for Video Recognition**
CLIPSitu-Vid utilizes **CLIP embeddings** to extract **rich visual-linguistic representations** from **individual video frames**. These embeddings are fed into a **temporal attention-based transformer**, which:
1. **Aggregates multi-frame context** using **self-attention**.
2. **Predicts structured outputs** (verb, roles, objects) at each timestamp.
3. **Uses learned constraints** to improve noun-role assignments.


## 🔬 VLM (VILA) Evaluation for Comparison

To compare **CLIPSitu-Vid** with a **Vision-Language Model (VLM)**, we use **VILA** for **zero-shot prompting** in Jupyter notebooks.  

📄 **VILA is only used as a benchmark** and is **not integrated into the main model**.

---

## 🏆 Acknowledgments

This project builds upon:
- **[CLIPSitu](https://github.com/LUNAProject22/CLIPSitu)** (leveraging **CLIP for image-based situation recognition**).
- **[VidSitu](https://github.com/TheShadow29/VidSitu)** (large-scale **video semantic role labeling** dataset).
- **[GVSR](https://github.com/zeeshank95/GVSR)** (**Graph-based Video Situation Recognition**).

We thank the authors of these projects for their foundational contributions.

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{CLIPSituVid2024,
  title={Effectively Leveraging CLIP for Generating Situational Summaries of Images and Videos
},
  author={Verma, D. and others},
  journal={Journal TBD},
  year={2024},
  url={https://arxiv.org/pdf/2407.20642}
}

@article{VidSitu2021,
  title={VidSitu: A Large-Scale Video Dataset for Situation Recognition},
  author={Sadhu, A. and others},
  journal={CVPR},
  year={2021}
}

@article{GVSR2021,
  title={GVSR: A Graph-Based Model for Video Situation Recognition},
  author={Zeeshan, K. and others},
  journal={arXiv},
  year={2021}
}
```
## 💡 Future Work
- Fine-tuning CLIP-based transformers for multi-frame attention.
- Logic-based constraints to improve role assignments.
- Multi-modal learning: Incorporating audio cues for enhanced understanding.

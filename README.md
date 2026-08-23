# Chagas-JEPA

### ECG-Based Chagas Disease Detection with a JEPA Foundation Model

Chagas-JEPA is a deep learning pipeline for detecting **Chagas disease from ECG signals**. The project adapts a pretrained **ECG-JEPA** representation-learning model to a binary Chagas classification task and explores whether a **Mixture-of-Layers (MoL)** aggregation module can improve downstream classification by using information from multiple transformer layers instead of relying only on the final representation.

The repository includes the model implementation, preprocessing and downstream-training code, saved fine-tuned checkpoints, and a **Streamlit web application** for testing ECG records through a simple upload interface.

> **Important:** This project is a research prototype and is **not a clinical diagnostic tool**. Predictions should not be used for medical diagnosis or treatment decisions.

---

## Overview

Chagas disease can produce cardiac abnormalities that are observable in ECG recordings, making ECG-based screening a useful research direction when more specialised testing is difficult to access. The goal of this project was to build a practical ECG-based classifier that can work with a **reduced 8-lead input** derived from a standard 12-lead ECG.

The core approach combines:

- **ECG-JEPA** — a Joint-Embedding Predictive Architecture adapted to ECG signals found from this GitHub repository:
- (https://github.com/sehunfromdaegu/ECG_JEPA), Linked paper: (https://arxiv.org/abs/2410.08559)
- **Transformer-based representation learning** — ECGs are split into patches and processed using self-attention to learn relationships across leads and time.
- **Self-supervised pretraining** — the encoder learns general ECG representations before downstream classification.
- **Mixture-of-Layers (MoL)** — intermediate transformer representations are treated as complementary sources of information and combined using a learned gating network. The work presented in (https://arxiv.org/abs/2509.00102) experiments with different multi-layer aggregation architectures and compares their effectiveness for ECG foundation models.
- **Downstream classification** — a classifier head is trained for binary Chagas-positive / Chagas-negative prediction.
- **Streamlit inference UI** — ECG records can be uploaded as WFDB `.dat` + `.hea` pairs and passed through the same preprocessing pipeline used by the trained model.

---

## Model Architecture

### 1. ECG-JEPA foundation model

The underlying model follows the ECG-JEPA architecture and uses a transformer encoder to learn ECG representations. The implementation in this repository is configured around:

- **8 ECG leads** for the reduced-lead downstream task
- **2,500 time samples** per input after resampling
- **50-sample patches** per lead
- **12 transformer encoder layers**
- **768-dimensional encoder embeddings**
- **16 attention heads**

The ECG-JEPA design uses **Cross-Pattern Attention (CroPA)** so that the model can learn relationships between patches within individual leads as well as relationships across leads at corresponding time positions.

The pretraining stage is self-supervised: rather than requiring disease labels for every ECG, the model learns useful ECG representations that can later be adapted to downstream tasks.

### 2. Mixture-of-Layers aggregation

A standard downstream classifier often uses the representation produced by only the final transformer layer. This project instead implements a **Mixture-of-Layers** module so that information from multiple encoder layers can contribute to the final representation.

The implementation consists of:

1. Hidden states from the transformer encoder layers are collected.
2. A small **gating network** produces a weight for each layer.
3. A softmax converts the weights into a learned mixture over the 12 layers.
4. The weighted layer representations are fused.
5. Token-level representations are pooled and passed through a projection block before the classification head.

The gating network is implemented as a small MLP with a **128-dimensional hidden layer**, followed by dropout and a 12-way output corresponding to the encoder layers.

This allows the model to learn which transformer layers provide the most useful representations for the downstream Chagas classification task.

### 3. Downstream classification

The repository supports both the original linear-evaluation style workflow and fine-tuning. The final Chagas pipeline uses a binary classification head to output the probability of a positive Chagas prediction.

The inference threshold in the Streamlit application is **0.5**:

- `1` → Chagas Positive
- `0` → Chagas Negative

---

## ECG Preprocessing

The Streamlit inference pipeline expects standard **12-lead WFDB recordings** and converts them into the input format expected by the trained model.

The pipeline is:

```text
12-lead ECG (.dat + .hea)
        │
        ▼
      WFDB
        │
        ▼
  Select 8 leads
  I, II, V1–V6
        │
        ▼
 Resample to 2,500 samples
        │
        ▼
 Per-lead Z-score normalization
        │
        ▼
  [batch, 8, 2500]
        │
        ▼
 Chagas-JEPA classifier
```

The reduced lead set is created directly in the application by retaining the first two standard leads and V1–V6. Inputs that do not have 2,500 samples are resampled before inference.

---

## Datasets

The project works with publicly available ECG datasets used across pretraining and downstream experiments, including:

| Dataset | Role in the project |
| --- | --- |
| **CODE-15%** | ECG representation learning / Chagas-related training data |
| **PTB-XL** | Downstream ECG classification and Chagas-related training/evaluation |
| **SaMi-Trop** | Chagas-related ECG training/evaluation |

The underlying ECG-JEPA implementation also contains support for additional ECG datasets used during development and benchmarking.

---

## Results

The project evaluates the Chagas classifier using standard classification metrics including **accuracy, precision, recall, F1-score, AUROC, and confusion matrices**.

The final evaluation reported in the project compared the MoL-enabled and non-MoL configurations:

| Configuration | Accuracy | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| **MoL Enabled** | 0.9949 | 0.9636 | 0.9578 | 0.9695 |
| **MoL Disabled** | 0.9962 | 0.9726 | 0.9657 | 0.9756 |

These results are included to document the experimental comparison; they should not be interpreted as clinical validation.

---

## Streamlit Demo

The repository contains a Streamlit interface in `ChagasDemo.py`.

The application provides three modes:

- **MoL Comparison** — runs both configurations and displays their predictions side by side.
- **MoL Enabled** — uses the MoL configuration.
- **MoL Disabled** — uses the non-MoL configuration.

The uploader expects the two WFDB files belonging to the same ECG record:

```text
record_name.dat
record_name.hea
```

After uploading both files, click **Run Prediction** to process the ECG and display the Chagas prediction and confidence.

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/inuksperera/ChagasDetectionECG.git
cd ChagasDetectionECG
```

### 2. Create a virtual environment

Python **3.10+** is recommended.

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit streamlit-option-menu
```

Install a compatible **PyTorch 2.0+** build for your machine (CPU or CUDA) if it is not already installed. For CUDA, use the installation command appropriate for your CUDA version from the official PyTorch instructions.

### 4. Check the fine-tuned weights

The trained checkpoints are stored in:

```text
FINETUNED_WEIGHTS/
```

The application expects these files:

```text
FINETUNED_WEIGHTS/
├── ejepa.pth
└── checkpoint_linear_eval_combined_data_20260415-192106.pth
```

`ejepa.pth` is used for the **MoL-enabled** configuration, while `checkpoint_linear_eval_combined_data_20260415-192106.pth` is used for the **MoL-disabled** configuration.

> The large checkpoint files were removed from the project archive provided for inspection here because of upload size limitations. In the GitHub repository, the `FINETUNED_WEIGHTS` folder should contain the trained weight files above.

### 5. Launch the Streamlit application

From the **root directory of the repository**, run:

```bash
streamlit run "ChagasDemo.py"
```

Streamlit will open the application in your browser.

### 6. Test the model using the included ECG samples

Sample testing records are already included in:

```text
testing data/
├── negative/
└── positive/
```

Each record consists of a `.dat` and `.hea` file with the same base name. For example:

```text
testing data/negative/20000_hr.dat
testing data/negative/20000_hr.hea
```

or:

```text
testing data/positive/3629.dat
testing data/positive/3629.hea
```

In the Streamlit application:

1. Select **MoL Comparison**, **MoL Enabled**, or **MoL Disabled** from the sidebar.
2. Upload **both** the `.dat` and `.hea` files for one test record.
3. Click **Run Prediction**.
4. The application will load the ECG through WFDB, reduce it to 8 leads, resample it, normalize the leads, and pass it to the trained model.

The included positive and negative examples make it possible to verify the complete inference pipeline without downloading the original research datasets.

---

## Repository Structure

```text
ChagasDetectionECG/
│
├── ChagasDemo.py                    # Streamlit inference application
├── detect_disease.py                # Model loading and inference
├── ecg_jepa.py                      # ECG-JEPA architecture
├── models.py                         # Encoder construction/loading
├── mol.py                            # Mixture-of-Layers implementation
├── linear_probe_utils.py             # Linear / fine-tuning classifier utilities
├── ecg_data.py                      # ECG data utilities and normalization
├── ptbxl_utils.py                   # PTB-XL data processing
├── ptbxl_chagas_utils.py            # Chagas/PTB-XL processing utilities
├── samitrop_utils.py                # SaMi-Trop data utilities
├── augmentation.py                  # Data augmentation utilities
│
├── downstream_tasks/
│   ├── finetuning.py                # Fine-tuning pipeline
│   ├── linear_eval.py               # Linear evaluation pipeline
│   └── output/                      # Development outputs/logs
│
├── configs/
│   └── downstream/                 # Training/evaluation configurations
│
├── FINETUNED_WEIGHTS/               # Trained model checkpoints
│
├── testing data/
│   ├── negative/                    # Example negative ECG records
│   └── positive/                    # Example positive ECG records
│
├── pretrain_ECG_JEPA.py             # Self-supervised pretraining pipeline
├── prepare_samitrop_data.py         # SaMi-Trop preparation
├── test.py                           # Inference/testing script
├── test2.py                          # Additional inference test
├── requirements.txt
└── README.md
```

---

## Training Pipeline

The broader training workflow is organised into two main stages:

### Stage 1 — Self-supervised pretraining

The ECG-JEPA encoder learns ECG representations from ECG data without requiring the downstream Chagas label for every training example.

```text
Large-scale ECG data
        │
        ▼
Patch ECG signals by lead/time
        │
        ▼
Cross-Pattern Attention
        │
        ▼
Transformer encoder
        │
        ▼
Learn general ECG representations
```

### Stage 2 — Downstream Chagas classification

The pretrained encoder is adapted to the Chagas task using a classification head. The MoL experiment adds the layer-aggregation module between the encoder representation and classifier.

<img width="2127" height="1855" alt="model diagram final" src="https://github.com/user-attachments/assets/d5d7646c-b079-4539-ab02-ae1f8bf9b115" />

---

## Key Files

### `ChagasDemo.py`
Streamlit application used for interactive inference. Handles file uploads, WFDB reading, preprocessing, model selection, and displaying predictions.

### `detect_disease.py`
Central inference function. Loads the trained checkpoint, constructs the classifier wrapper, prepares the input tensor, performs inference, and returns the predicted class and probability.

### `ecg_jepa.py`
Contains the ECG-JEPA transformer architecture, including the encoder, predictor, patch/token processing, positional embeddings, masking, and Cross-Pattern Attention mechanism.

### `mol.py`
Implements the Mixture-of-Layers architecture through the `MoLJEPA` wrapper and the `PMA` gating/aggregation module.

### `downstream_tasks/finetuning.py`
Fine-tuning pipeline for adapting the pretrained encoder to downstream ECG classification tasks. It supports the `mol_jepa` configuration and saves combined model checkpoints.

### `downstream_tasks/linear_eval.py`
Linear-evaluation pipeline in which the encoder representation is used with a classifier head.

---

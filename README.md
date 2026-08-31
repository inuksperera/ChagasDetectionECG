# Chagas-JEPA

### ECG-Based Chagas Disease Detection with a JEPA Foundation Model

Chagas-JEPA is a deep learning pipeline for detecting **Chagas disease from ECG signals**. The project adapts a pretrained **ECG-JEPA** representation-learning model to a binary Chagas classification task and explores whether a **Mixture-of-Layers (MoL)** aggregation mechanism can improve downstream classification by using information from multiple transformer layers instead of relying only on the final layer.

The repository includes the model implementation, preprocessing and downstream-training code, saved fine-tuned checkpoints, and a **Streamlit web application** for testing ECG records through a simple drag-and-drop upload interface.

---

## Overview

Chagas disease can produce cardiac abnormalities that are observable in ECG recordings, making ECG-based screening a useful research direction when more specialised testing is difficult to access. The goal of this project was to build a practical ECG-based classifier that can work with a **reduced 8-lead input** derived from a standard 12-lead ECG.

The core approach combines:

- **ECG-JEPA** — a model that implemented the Joint-Embedding Predictive Architecture and adapted it to work with ECG signals found from this GitHub repository that also initiated an experimentation towards using a reduced number of leads:
(https://github.com/sehunfromdaegu/ECG_JEPA), paper linked to GitHub repo: (https://arxiv.org/abs/2410.08559)
- **Mixture-of-Layers (MoL)** — intermediate transformer layers are treated as rich sources of information and combined using a learned gating network. The work presented in (https://arxiv.org/abs/2509.00102) experiments with different multi-layer aggregation architectures and compares their effectiveness for ECG foundation models.
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
Break the dataset into batches
  [batch, 8, 2500]
        │
        ▼
 Chagas-JEPA encoder
        │
        ▼
  Gating Network
        │
        ▼
  Weighted sum
        │
        ▼
 Classifier head
        │
        ▼
Chagas Prediction
```

The reduced lead set is created directly in the application by retaining the first two standard leads and V1–V6. Inputs that do not have 2,500 samples are resampled before inference.

---

### Downstream Chagas classification

The following image represents the architecture diagram for the model. The pretrained encoder is trained to learn general ECG features by using a masking mechanism. It is then trained on Chagas-specific data to learn abnormalities that arise in a patient that has contracted the disease. Disease prediction is conducted using a specified threshold in the classification head. The MoL mechanism is implemented by adding a gating network and a weighted sum between the encoder representation and classifier.

<img width="2127" height="1855" alt="model diagram final" src="https://github.com/user-attachments/assets/d5d7646c-b079-4539-ab02-ae1f8bf9b115" />

---

## Datasets

The project works with publicly available ECG datasets used across pretraining and downstream experiments, including:

| Dataset | Role in the project |
| --- | --- |
| **CODE-15%** | General ECG representation learning |
| **PTB-XL** | Downstream ECG classification and Chagas-related training/evaluation |
| **SaMi-Trop** | Downstream ECG classification and Chagas-related training/evaluation |

---

## Demo Video 
 
A short demonstration of the Chagas-JEPA Web Application is available at the link below. It shows how ECG samples are uploaded to the interface and processed to generate Chagas disease predictions using the different model configurations. 
 
[▶️ Demo Video](https://drive.google.com/file/d/1tlyiByr0N6MXi9-Bt6rJRavcIRObrkjX/view?usp=sharing)

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
pip install -r new_requirements.txt
pip install streamlit streamlit-option-menu
```

Install a compatible **PyTorch 2.0+** build for your machine (CPU or CUDA) if it is not already installed. For CUDA, use the installation command appropriate for your CUDA version from the official PyTorch instructions.

### 4. Download and add the fine-tuned weights

The fine-tuned model weights are available for download in the following Google Drive folder. Download the `FINETUNED_WEIGHTS` folder from it.
**Note: the `FINETUNED_WEIGHTS` folder is around 600MB, so it may take some time for Google Drive to zip the contents and start the download.**
(https://drive.google.com/drive/folders/1q89TbubEyNSlgJsxxmO85GLUVbYreoft?usp=sharing)

The folder will be downloaded as a `.zip` file. After downloading:
1. Extract the `.zip` file.
2. Place the extracted `FINETUNED_WEIGHTS` folder directly in the **root directory of the repository**.

The final structure should look like:

```text
ChagasDetectionECG/
├── FINETUNED_WEIGHTS/
│   ├── ejepa.pth
│   └── checkpoint_linear_eval_combined_data_20260415-192106.pth
├── ChagasDemo.py
├── 
```


### 5. Launch the Streamlit application

From the **root directory of the repository**, run:

```bash
streamlit run "ChagasDemo.py"
```

Streamlit will open the application in your browser.

### 6. Test the model using the Streamlit app and included ECG samples

The repository contains sample ECG records in:

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

The Streamlit application provides three modes:

MoL Enabled - uses the weights that were trained **with** the MoL configuration.
MoL Disabled - uses the weights that were trained **without** the MoL configuration.
MoL Comparison - runs both configurations and displays their predictions side by side.


To test the model, select a mode from the sidebar and upload both the .dat and .hea files belonging to the same ECG record. Then click Run Prediction. The application will load the ECG through WFDB, reduce it to 8 leads, resample and normalize the signal, and pass it through the selected model to display the predicted Chagas classification and confidence.

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
Fine-tuning pipeline that trains the entire model pipeline, including the pretrained encoder and classifier, to adapt the model to downstream ECG classification tasks.

### `downstream_tasks/linear_eval.py`
Linear-evaluation pipeline that keeps the encoder frozen and trains only the classifier head, providing a faster approach that is better suited for development and evaluation.

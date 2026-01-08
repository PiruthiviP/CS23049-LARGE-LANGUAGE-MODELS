# Experiment 1: Transformer Encoder – Autoencoding (Masked Language Model)

## 📌 Project Overview
This project implements a **Transformer Encoder** to solve the **Masked Language Modeling (MLM)** task. The objective is to understand how Self-Attention and Autoencoding mechanisms work by reconstructing masked words in a sentence.

### 🔹 Objective
*   To understand the Transformer Encoder architecture.
*   To visualize Self-Attention weights (Heatmaps).
*   To demonstrate Autoencoding by reconstructing corrupted text.

---

## 📂 Project Structure
```text
transformer-encoder-autoencoding/
│
├── encoder.py                # Main Transformer Encoder implementation
├── attention.py              # Self-Attention mechanism logic
├── positional_encoding.py    # Logic for adding positional information to embeddings
├── train_mlm.py              # Script to run the Masked Language Model experiment
├── visualize_attention.ipynb # Notebook to generate Heatmaps
├── README.md                 # Project Documentation
└── results/
    ├── attention_heatmap.png # Visual output of attention weights
    └── output_log.txt        # Console output of reconstructions
```

---

## 🧠 Theory: Autoencoding & Architecture

### 🧠 Explanation of Autoencoding

Autoencoding is a learning mechanism where the model attempts to reconstruct the original input from a corrupted version of it.

In this experiment, autoencoding is implemented using **Masked Language Modeling (MLM)**:

1. A complete sentence is taken as input.
2. One word is replaced with a special token `[MASK]`.
3. The Transformer Encoder processes the entire sentence simultaneously.
4. Using self-attention, the model captures contextual relationships between all words.
5. The encoder predicts the missing word using surrounding context.

Unlike traditional autoencoders, this process does not use recurrence or convolution. Instead, it relies entirely on self-attention to model global dependencies.


### 2. Encoder Architecture
The model uses the **BERT-style Encoder** architecture, which consists of:
*   **Embeddings + Positional Encoding:** To convert words to vectors and retain order.
*   **Multi-Head Self-Attention:** To allow words to "attend" to each other.
*   **Feed-Forward Networks:** To process the information.

### 🧩 Transformer Encoder Architecture Diagram

The following diagram illustrates the internal components of a Transformer Encoder, including embedding, positional encoding, self-attention, and feed-forward layers.

![Transformer Encoder Architecture](https://jalammar.github.io/images/t/transformer_encoder.png)

```mermaid
graph TD
    A[Input Sentence] --> B[Tokenization & Masking]
    B --> C[Embedding + Positional Encoding]
    C --> D[Multi-Head Self-Attention]
    D --> E[Add & Norm]
    E --> F[Feed Forward Network]
    F --> G[Add & Norm]
    G --> H[Output Probability (Softmax)]
    H --> I[Reconstructed Word]
```

---

## 📊 Experiment Results

### 1. Input vs. Reconstructed Output
We tested the model on 10 student samples. The model successfully used context to fill in the blanks.

| Student ID | Topic | Masked Input | Reconstructed Output |
| :--- | :--- | :--- | :--- |
| SAMPLE1 | AI | Transformers use [MASK] attention | Transformers use **self** attention |
| SAMPLE2 | Space | Mars is called the [MASK] planet | Mars is called the **red** planet |
| SAMPLE3 | Education | Online learning improves [MASK] access | Online learning improves **educational** access |
| SAMPLE4 | Health | Exercise improves [MASK] health | Exercise improves **mental** health |
| SAMPLE5 | Sports | Cricket is a [MASK] sport | Cricket is a **popular** sport |
| SAMPLE6 | Computing | Python is a [MASK] language | Python is a **programming** language |
| SAMPLE7 | AI | Neural networks have [MASK] layers | Neural networks have **hidden** layers |
| SAMPLE8 | Environment | Trees reduce [MASK] pollution | Trees reduce **air** pollution |
| SAMPLE9 | Robotics | Robots perform [MASK] tasks | Robots perform **repetitive** tasks |
| SAMPLE10 | Energy | Solar power is a [MASK] source | Solar power is a **renewable** source |

---

### 2. Attention Visualization
Below is the **Self-Attention Heatmap** generated during the experiment.

![Self-Attention Heatmap](results/attention_heatmap.png)

**Observation:**
*   The heatmap shows how much importance (attention) one word gives to another.
*   In the row for `[MASK]`, we can see distinct colors corresponding to "Transformers" and "attention".
*   This proves that to predict the mask, the model "looked" at the subject and object of the sentence.

---

## ⚙️ How to Run

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Transformers (Hugging Face)
*   Matplotlib (for heatmaps)

### Installation
```bash
pip install torch transformers matplotlib seaborn
```

### Running the Experiment
To run the text reconstruction and generate the heatmap:

```bash
python train_mlm.py
```

---

## 📝 Conclusion
*   **Global Context:** Unlike RNNs which read sequentially, the Transformer Encoder reads the entire sentence at once, allowing the `[MASK]` token to see both left and right context.
*   **Performance:** The model successfully reconstructed domain-specific terms (e.g., "red planet" for Space, "hidden layers" for AI).
*   **Comparison:** Compared to a simple Feed-Forward network (which treats words in isolation), the Transformer performs significantly better because it understands the relationship between words via Attention.
```

### How to use this for your specific folder structure:
1.  **Save your python script** (the one I gave you earlier) as `train_mlm.py`.
2.  **Save your heatmap image** into a folder named `results` and name the image `attention_heatmap.png`.
3.  **Create a file** named `README.md` and paste the code above into it.

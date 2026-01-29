# Transformer Autoregression Project

This project implements a Transformer-based sequence-to-sequence model for autoregressive tasks. It demonstrates the core components of the Transformer architecture, including Encoders, Decoders, and Attention mechanisms, applied to simple language tasks like Paraphrasing, Q&A, and Text Generation.

## Project Structure

The codebase is organized into modular components:

### Core Components
- **`transformer.py`**: The main `TransformerSeq2Seq` class that combines the Encoder and Decoder.
- **`encoder.py`**: Implements the Encoder module. Current implementation features a simplified embedding and linear projection layer.
- **`decoder.py`**: Implements the Decoder module with Causal Masking and Self-Attention (or Multi-head Attention).
- **`attention_masks.py`**: Utility to generate causal masks (triangular matrices) to prevent the model from peeking at future tokens during training/inference.
- **`model.py`**: An alternative/standalone implementation of the Transformer model (Self-contained).

### Workflows
- **`train.py`**: 
  - Trains the model on a small, hardcoded dataset of sentence pairs.
  - Uses a word-level tokenizer.
  - Saves the trained model weights to `seq2seq.pt`.
- **`inference.py`**:
  - A script for running inference with the trained model.
  - Includes an interactive loop to generic text responses.
  - *Note*: Contains its own class definitions and vocabulary handling (character-level vs word-level), which may differ from the training script.

## Installation & Dependencies

This project requires **PyTorch**.

```bash
pip install torch
```

## How to Run

### 1. Training the Model
To train the model on the sample dataset:
```bash
python train.py
```
This will output the training loss per epoch and save the final model checkpoints to `seq2seq.pt`.

### 2. Running Inference
To interact with the model:
```bash
python inference.py
```
You can type sentences or questions to see the model's autoregressive output.

## Implementation Details

- **Dataset**: A small set of synthetic pairs (e.g., "AI improves healthcare" -> "AI enhances medical diagnosis and treatment").
- **Tokenizer**: 
  - `train.py`: Basic whitespace-based word tokenization.
  - `inference.py`: Character-based tokenization (experimental).
- **Architecture**: A simplified Transformer variant focusing on the sequence-to-sequence flow.

## Assignment Notes
- The project demonstrates the logic of `Encoder -> Decoder` information flow.
- "Autoregression" is highlighted in the generation phase, where the model predicts one token at a time, feeding it back as input for the next step.

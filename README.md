# Speech Emotion Recognition (SER) with Wav2Vec2 and QLoRA

This is a part of project that builds a **Speech Emotion Recognition (SER)** system using **Wav2Vec2** models.  
It focuses on training and fine-tuning large pre-trained models efficiently using **QLoRA** (Quantized Low-Rank Adaptation).

We use emotional speech datasets like **CREMA-D**, **RAVDESS**, and **TESS**, and aim to classify different emotions (such as happy, sad, angry, fear, etc.) based on audio recordings.

There are two main approaches in this project:
- **Fine-tune a Wav2Vec2 model with a custom classification head**
- **Apply QLoRA on a 4-bit quantized Wav2Vec2 model to save memory and speed up training**

The project handles:
- Dataset preprocessing
- Audio feature extraction
- Model fine-tuning
- Training with QLoRA
- Model evaluation and metrics reporting

---

## Project Highlights

- **Speech Audio Input**: Works directly with raw `.wav` files.
- **Emotion Categories**: Disgust, Happy, Sad, Fear, Neutral, Angry, Surprised.
- **Model Architecture**:
  - Pre-trained **Wav2Vec2-Large-XLSR-53** backbone
  - Added lightweight classification head
  - LoRA adapters applied only to key transformer layers
- **QLoRA Training**:
  - Quantized model to 4-bit (NF4 format)
  - Fine-tuned with LoRA for memory efficiency
- **Evaluation**:
  - Accuracy, UAR (Unweighted Average Recall), F1-score, Cohen's Kappa
  - Confusion Matrix Visualization
- **Model Saving**: Best model checkpoints saved for later use

---

## Why QLoRA?

Training full-size Wav2Vec2 models can be very memory-intensive.  
**QLoRA** helps by:
- Reducing model size dramatically (4-bit quantization)
- Training only small LoRA adapters instead of the full model
- Achieving competitive performance even on limited hardware (e.g., a single GPU)

---

## How to Use - using "ICE - PACE"

1. Download the emotional speech datasets (CREMA-D, RAVDESS, TESS).
2. Preprocess the metadata and audio files.
3. Install required Python packages.
4. Choose to:
   - Fine-tune Wav2Vec2 normally, or
   - Fine-tune with QLoRA for more efficient training.
5. Evaluate model performance.

All training and evaluation scripts are provided!

---

## Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes (for 4-bit quantization)
- scikit-learn
- torchaudio
- librosa
- matplotlib

---


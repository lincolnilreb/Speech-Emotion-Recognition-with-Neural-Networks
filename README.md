# Speech Emotion Recognition (SER) with Wav2Vec2 and QLoRA


## 📖 Project Overview  
This project develops a **Speech Emotion Recognition (SER)** system by fine-tuning large pre-trained speech models, primarily **Wav2Vec2** and **Whisper**, on benchmark emotional audio datasets. The system classifies emotions directly from raw `.wav` files, leveraging **parameter-efficient fine-tuning** techniques (LoRA / QLoRA) to balance performance with computational efficiency.  

The pipeline covers end-to-end SER: dataset preprocessing, feature extraction, model adaptation, training, and evaluation with industry-standard metrics.  


## ✨ Key Highlights  
- **10,000+ audio samples** across benchmark datasets: CREMA-D, RAVDESS, and TESS.  
- **Fine-tuned large speech models** (Wav2Vec2, Whisper) with a custom classification head.  
- Achieved **84% classification accuracy** across seven core emotion categories (disgust, happy, sad, fear, neutral, angry, surprised).  
- Applied **LoRA & QLoRA** for efficient fine-tuning:  
  - 4-bit quantization (NF4 format).  
  - Reduced training cost and memory footprint while sustaining model accuracy.  
- Comprehensive evaluation with **Accuracy, UAR (Unweighted Average Recall), F1-score, Cohen’s Kappa**, and confusion matrices.  


## 🛠️ Tech Stack  
- **Frameworks & Libraries**: PyTorch · Hugging Face Transformers · PEFT · BitsAndBytes  
- **Data & Audio Processing**: torchaudio · librosa · scikit-learn · matplotlib  
- **Techniques**:  
  - Transfer Learning with Transformers  
  - LoRA / QLoRA for parameter-efficient fine-tuning  
  - Model quantization & optimization  
- **Environments**:  
  - ICE-PACE HPC cluster  
  - Single-GPU training compatibility  

## 🚀 Impact  
- This project demonstrates how **state-of-the-art speech transformers** can be adapted for emotion recognition in a **resource-constrained setting**.  
- By applying **QLoRA**, the system maintains high performance while cutting GPU memory usage and training costs—making advanced SER solutions accessible even outside large compute environments.  



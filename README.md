# Real-Time Greek Sign Language Recognition and Translation

![Live Translation Screenshot](live_mobile.webp)

A complete, end-to-end AI system for real-time recognition of the Greek fingerspelling alphabet (static signs) and Greek words/phrases (dynamic signs), using computer vision and deep learning techniques.

---

## Project Structure

The project consists of two independent subsystems, each optimized for the nature of its data:

1. **Greek Alphabet Recognition:** Classification of 24 static letters using machine learning.
2. **Greek Phrases Recognition:** Recognition of temporal sequences (words) using deep learning architectures, with a FastAPI web server for live streaming.

### Technology Stack
* **Computer Vision:** OpenCV, MediaPipe (spatial feature / landmark extraction).
* **Machine Learning (Static Signs):** Multi-Layer Perceptron (MLP).
* **Deep Learning (Dynamic Signs):** LSTM, BiLSTM, Transformer, SPOTER.
* **Backend & Deployment:** Python, FastAPI, Cloudflare Tunnels.

---

## Installation and Prerequisites

Python 3.10 or later and a webcam are required.

**1. Clone the repository:**
```bash
git clone https://github.com/CharalamposZisis/Greek-Sign-Language.git
cd Greek-Sign-Language
```

**2. Create a Virtual Environment (optional but recommended):**
```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For Linux/macOS:
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Usage Instructions (Real-Time Inference)

### System 1: Static Letter Recognition
This subsystem activates the webcam and recognizes fingerspelling letters in real time, displaying a confidence score using the best-performing trained MLP model.

**Run:**
```bash
cd "Greek Alphabet Recognition"
python live_recognition_relative.py
```

### System 2: Dynamic Word Recognition (Live Server)
This subsystem runs the SPOTER (Sign Pose-based Transformer) model through a FastAPI server, enabling live, real-time translation.

**Start the Local Server:**
```bash
cd "Greek Phrases Recognition"
python server.py
```

The server will start locally. Open [the local translation interface](http://localhost:8000) in your browser.

**Public Access (via Cloudflare):**

To share the live translation stream with others over the internet, open a second terminal and run:
```bash
cloudflared tunnel --url http://localhost:8000
```

A temporary public URL will be generated, which you can share.

---

## Model Performance

The final production models were selected following extensive evaluation and checks for overfitting.

Due to the lack of an open dataset for the Greek fingerspelling alphabet, a dedicated Python tool was developed and a custom dataset was created from scratch.

### Static Signs (Letters)
Random Forest, KNN, SVM, and MLP algorithms were evaluated.

* **Selected Model:** Multi-Layer Perceptron (MLP).
* **Accuracy (Test Set):** 99.01%.
* **Inference Time:** 0.0025 ms / frame (preventing flickering).

![Model Accuracy Comparison](Greek%20Alphabet%20Recognition/model_comparison_accuracy.png)

### Inference Time Comparison

The following chart compares inference times across the evaluated models. MLP recorded the lowest time at **0.0025 ms per sample**.

![Model Inference Time Comparison](Greek%20Alphabet%20Recognition/model_comparison_time.png)

### Dynamic Signs (Words)
A unified, hybrid dataset of 5,864 samples was created to train the dynamic models.

* 5,064 samples were obtained from the public GSL dataset.
* An additional 800 samples were collected specifically for this project by the authors using a webcam.
* These custom samples were combined with the GSL dataset, increasing the total training sample size.

![Greek Word Data Collection Process](Picture5.png)

LSTM, BiLSTM, Transformer, and SPOTER architectures were evaluated.

* **Evaluated Models:** BiLSTM achieved 97.60% accuracy, LSTM achieved 97.21%, and the Transformer Encoder achieved 96.21%.
* **Selected Model:** SPOTER (Sign Pose-based Transformer) demonstrated excellent performance, achieving 99.77% accuracy. Its high performance is attributed to its architecture, which combines an encoder and a decoder with a class query mechanism.

---

## Future Extensions (Smart Glasses)

A proposed extension involves integrating the system into smart glasses with a transparent OLED display. The glasses would capture video through a small built-in camera, run the SPOTER algorithm on a processor such as a Raspberry Pi, and display predictions directly on the small OLED screen. This would enable fully standalone operation, allowing users to view translations in real time without needing a smartphone or computer screen.

![Smart Glasses with a Built-in Camera and OLED Display](Picture6.png)

---

## Development Team

This research project was carried out by:

* **[Charalampos Zisis](https://github.com/CharalamposZisis)**
* **Christos Tsamis**

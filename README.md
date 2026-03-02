
# 🎥 VidScribe AI

VidScribe is an AI-powered web application that converts audio and video files into structured, searchable knowledge. 
It goes beyond simple transcription by applying advanced Natural Language Processing (NLP) techniques such as 
summarization, question answering, sentiment analysis, and semantic similarity comparison.

---

## 🚀 Features

- 🎙 Audio & Video Transcription (AssemblyAI Integration)
- 🧠 AI-Based Summarization (Transformer Models)
- ❓ Contextual Question Answering
- 😊 Sentiment Analysis
- 📊 Word Cloud Generation
- 🔎 Semantic Transcript Comparison
- 💬 Interactive Chat Interface
- ⚡ Asynchronous Background Task Processing

---

## 🏗 System Overview

VidScribe follows a multi-stage AI pipeline:

1. User uploads audio/video file  
2. Speech-to-text transcription  
3. Transcript storage  
4. AI processing:
   - Summarization
   - Question Answering
   - Sentiment Analysis
   - Semantic Similarity
5. Results displayed via web interface

---

## 🛠 Tech Stack

### Backend
- Python
- Flask
- PyTorch
- HuggingFace Transformers
- Sentence Transformers
- AssemblyAI API

### Frontend
- HTML
- CSS
- JavaScript
- Jinja2 Templates

---

## 📂 Project Structure

```
VidScribe/
│
├── app.py
├── requirements.txt
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── chatbot.js
│       └── transcript.js
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── transcribe.html
│   ├── transcript.html
│   └── transcription_status.html
│
└── uploads/
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```
git clone https://github.com/Pratham1708/VidScribe.git
cd VidScribe
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Set Environment Variable

Mac/Linux:
```
export ASSEMBLYAI_API=your_api_key
```

Windows:
```
set ASSEMBLYAI_API=your_api_key
```

### 5️⃣ Run Application

```
python app.py
```

Open browser at:
http://127.0.0.1:5000/

---

## 📌 Use Cases

- Lecture transcription & revision
- Meeting summarization
- Research discussions
- Podcast analysis
- Content indexing
- Media comparison

---

## 🔮 Future Improvements

- Chunk-based hierarchical summarization
- Retrieval-Augmented Generation (RAG)
- Speaker diarization
- Database-backed storage
- React frontend
- Mobile integration
- Performance dashboard

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Pratham Jindal  

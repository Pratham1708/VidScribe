# 🎥 VidScribe

**VidScribe** is an AI-powered Flask web application that transforms audio and video files into intelligent, structured insights. Beyond simple transcription, VidScribe enables summarization, question answering, sentiment analysis, and semantic comparison of transcripts through advanced NLP models.

It is designed as a multi-stage transcript intelligence system that converts unstructured speech into searchable, interactive knowledge.

---

## 🚀 Features

- 🎙 **Audio & Video Transcription**  
  Upload media files and automatically generate transcripts using speech-to-text integration.

- 🧠 **AI-Powered Summarization**  
  Generate concise summaries from long transcripts using transformer-based models.

- ❓ **Contextual Question Answering**  
  Ask questions about your transcript and receive context-aware answers.

- 😊 **Sentiment Analysis**  
  Analyze emotional tone across different parts of the transcript.

- 📊 **Word Cloud Generation**  
  Visualize key terms and frequently used words.

- 🔎 **Semantic Transcript Comparison**  
  Compare two transcripts and measure content similarity.

- 💬 **Interactive Chat Interface**  
  Chatbot-style assistant for guiding transcript analysis.

- ⚡ **Asynchronous Processing**  
  Background task handling for smooth user experience during AI processing.

---

## 🛠 Tech Stack

### Backend
- Python
- Flask
- Transformers (HuggingFace)
- PyTorch
- Sentence Transformers
- AssemblyAI (Speech-to-Text API)

### Frontend
- HTML5
- CSS3
- JavaScript
- Jinja2 Templating

---

## 📂 Project Structure


VidScribe/
│
├── app.py # Main Flask application
├── requirements.txt # Project dependencies
│
├── static/
│ ├── css/
│ │ └── style.css # Styling
│ └── js/
│ ├── chatbot.js # Chatbot UI logic
│ └── transcript.js # Transcript handling
│
├── templates/
│ ├── base.html # Base layout
│ ├── index.html # Home page
│ ├── transcribe.html # File upload page
│ ├── transcript.html # Transcript display
│ └── transcription_status.html # Status page
│
└── uploads/ # Uploaded media files


---

## ⚙️ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Pratham1708/VidScribe.git
cd VidScribe
2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Set Environment Variable

Set your AssemblyAI API key:

Mac/Linux

export ASSEMBLYAI_API=your_api_key

Windows

set ASSEMBLYAI_API=your_api_key
5️⃣ Run the Application
python app.py

Open your browser and visit:

http://127.0.0.1:5000/
📌 Use Cases

Lecture transcription & revision

Meeting intelligence

Research discussion analysis

Podcast summarization

Content indexing & comparison

🔮 Future Enhancements

Chunk-based hierarchical summarization

Retrieval-Augmented Question Answering (RAG)

Speaker diarization

Database-backed transcript storage

React-based frontend

Mobile application integration

Performance benchmarking dashboard

📄 License

This project is licensed under the MIT License.

🙌 Acknowledgements

Flask Documentation: https://flask.palletsprojects.com/

HuggingFace Transformers

AssemblyAI Speech-to-Text API

# VidScribe

**VidScribe** is a Flask-based web application that enables users to upload audio or video files and get them transcribed automatically. The app also features a chatbot-style interface to make interactions engaging and intuitive.

## 🔧 Features

- 🎙 Upload and transcribe audio/video files
- 💬 Chatbot-style UI for enhanced interaction
- 📜 Display transcription status and results
- 💡 Responsive interface using HTML/CSS/JS
- 🗂 Organized template and static file structure

## 🛠 Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Templating Engine:** Jinja2
- **Transcription Engine:** *(Custom speech-to-text integration expected)*

## 📂 Project Structure

```
VidScribe/
│
├── app.py                      # Main Flask application
├── static/
│   ├── css/
│   │   └── style.css           # Styling
│   └── js/
│       ├── chatbot.js          # Chatbot UI logic
│       └── transcript.js       # Transcript handling
│
├── templates/
│   ├── base.html               # Base layout
│   ├── index.html              # Home page
│   ├── transcribe.html         # File upload page
│   ├── transcript.html         # Transcript display
│   └── transcription_status.html # Status page
│
└── uploads/
    └── story1.mp4          # Upload sample media
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/VidScribe.git
cd VidScribe
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: If `requirements.txt` is missing, install Flask manually:
> ```bash
> pip install Flask
> ```

### 4. Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser.

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more information.

## 🙌 Acknowledgements

- Flask Documentation: https://flask.palletsprojects.com/


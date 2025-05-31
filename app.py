import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    pipeline,
    BartForConditionalGeneration,
    BartTokenizer
)
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import assemblyai as aai
import textwrap
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import uuid
import json
import threading
import queue
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import time

# Download NLTK data (first-time setup)
nltk.download('punkt', quiet=True)

# Configure API keys
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API")  # Replace with your actual key
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Configure Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'mp4', 'wav', 'avi', 'mkv', 'mov', 'wmv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Task Queue for threading
class TaskQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self._running = True
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.results = {}
    
    def _worker(self):
        while self._running:
            try:
                task_id, task, args = self.queue.get(timeout=0.5)
                try:
                    result = task(*args)
                    self.results[task_id] = {"status": "completed", "result": result}
                except Exception as e:
                    self.results[task_id] = {"status": "error", "error": str(e)}
                self.queue.task_done()
            except queue.Empty:
                continue
    
    def add_task(self, task, args=()):
        task_id = str(uuid.uuid4())
        self.results[task_id] = {"status": "processing"}
        self.queue.put((task_id, task, args))
        return task_id
    
    def get_result(self, task_id):
        return self.results.get(task_id, {"status": "not_found"})
    
    def stop(self):
        self._running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1)

class NeuralTranscriptProcessor:
    def __init__(self, device=None):
        # Set device (GPU if available, else CPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Caches
        self.transcript_cache = {}
        self.summary_cache = {}
        self.embeddings_cache = {}
        
    def _initialize_models(self):
        """Initialize all neural network models"""
        print("Loading neural network models...")
        
        # Summarization models
        self.summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        
        # Question Answering model
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(self.device)
        
        # Sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Sentence encoder for semantic similarity
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        print("All models loaded successfully!")
    
    def transcribe_audio(self, file_path=None, file_url=None):
        """Transcribe audio/video file using AssemblyAI"""
        if not aai.settings.api_key:
            return None, "Error: AssemblyAI API key is missing."
            
        if file_path in self.transcript_cache:
            return self.transcript_cache[file_path], "Loaded from cache"
            
        transcriber = aai.Transcriber()
        
        try:
            config = aai.TranscriptionConfig(language_code="en")
            if file_url:
                transcript = transcriber.transcribe(file_url, config=config)
            elif file_path and os.path.exists(file_path):
                transcript = transcriber.transcribe(file_path, config=config)
            else:
                return None, "Error: Please provide a valid file path or URL."
                
            if transcript.error:
                return None, f"Transcription error: {transcript.error}"
                
            self.transcript_cache[file_path] = transcript.text
            return transcript.text, "Transcription successful"
            
        except Exception as e:
            return None, f"An error occurred: {str(e)}"
    
    def summarization(self, text, max_length=150, min_length=30):
        """Generate summary using BART"""
        inputs = self.summarization_tokenizer(
            [text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        summary_ids = self.summarization_model.generate(
            inputs['input_ids'],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True
        )

        summary = self.summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def answer_question(self, context, question):
        """Answer questions using RoBERTa QA model"""
        inputs = self.qa_tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = self.qa_tokenizer.convert_tokens_to_string(
            self.qa_tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end]
            )
        )
        
        return answer
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text segments"""
        sentences = sent_tokenize(text)
        results = self.sentiment_analyzer(sentences)
        return list(zip(sentences, results))
    
    def generate_word_cloud(self, text):
        """Generate and save a word cloud visualization"""
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(text)
        
        # Save to a byte buffer
        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Convert to base64 for embedding in HTML
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return img_str
    
    def get_text_embedding(self, text):
        """Get embedding vector for a text"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
            
        embedding = self.sentence_encoder.encode(
            text,
            convert_to_tensor=True,
            device=self.device
        )
        self.embeddings_cache[text] = embedding
        return embedding
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_text_embedding(text1)
        emb2 = self.get_text_embedding(text2)
        return util.pytorch_cos_sim(emb1, emb2).item()

# Initialize processor and task queue
processor = NeuralTranscriptProcessor()
task_queue = TaskQueue()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Add transcription task to queue
            task_id = task_queue.add_task(processor.transcribe_audio, (file_path,))
            
            # Store task_id in session
            session['transcription_task_id'] = task_id
            session['file_path'] = file_path
            
            return redirect(url_for('transcription_status'))
    
    return render_template('transcribe.html')

@app.route('/transcription_status')
def transcription_status():
    task_id = session.get('transcription_task_id')
    
    if not task_id:
        flash('No transcription in progress')
        return redirect(url_for('transcribe'))
    
    result = task_queue.get_result(task_id)
    
    if result['status'] == 'completed':
        transcript, message = result['result']
        if transcript:
            session['transcript'] = transcript
            return redirect(url_for('transcript_view'))
        else:
            flash(f'Transcription failed: {message}')
            return redirect(url_for('transcribe'))
    elif result['status'] == 'error':
        flash(f'Error: {result["error"]}')
        return redirect(url_for('transcribe'))
    
    # Still processing
    return render_template('transcription_status.html')

@app.route('/transcript')
def transcript_view():
    transcript = session.get('transcript')
    
    if not transcript:
        flash('No transcript available')
        return redirect(url_for('transcribe'))
    
    return render_template('transcript.html', transcript=transcript)

@app.route('/summarize', methods=['POST'])
def summarize():
    transcript = session.get('transcript')
    
    if not transcript:
        return jsonify({'error': 'No transcript available'})
    
    task_id = task_queue.add_task(processor.summarization, (transcript,))
    
    return jsonify({'task_id': task_id})

@app.route('/summary_result/<task_id>')
def summary_result(task_id):
    result = task_queue.get_result(task_id)
    
    if result['status'] == 'completed':
        return jsonify({'status': 'completed', 'summary': result['result']})
    elif result['status'] == 'error':
        return jsonify({'status': 'error', 'error': result['error']})
    else:
        return jsonify({'status': 'processing'})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    transcript = session.get('transcript')
    question = request.json.get('question')
    
    if not transcript:
        return jsonify({'error': 'No transcript available'})
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    task_id = task_queue.add_task(processor.answer_question, (transcript, question))
    
    return jsonify({'task_id': task_id})

@app.route('/question_result/<task_id>')
def question_result(task_id):
    result = task_queue.get_result(task_id)
    
    if result['status'] == 'completed':
        return jsonify({'status': 'completed', 'answer': result['result']})
    elif result['status'] == 'error':
        return jsonify({'status': 'error', 'error': result['error']})
    else:
        return jsonify({'status': 'processing'})

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    transcript = session.get('transcript')
    
    if not transcript:
        return jsonify({'error': 'No transcript available'})
    
    # Limit to first 5000 chars for performance
    task_id = task_queue.add_task(processor.analyze_sentiment, (transcript[:5000],))
    
    return jsonify({'task_id': task_id})

@app.route('/sentiment_result/<task_id>')
def sentiment_result(task_id):
    result = task_queue.get_result(task_id)
    
    if result['status'] == 'completed':
        # Convert to JSON-serializable format
        sentiment_results = []
        for sentence, analysis in result['result']:
            sentiment_results.append({
                'sentence': sentence,
                'label': analysis['label'],
                'score': analysis['score']
            })
        return jsonify({'status': 'completed', 'results': sentiment_results})
    elif result['status'] == 'error':
        return jsonify({'status': 'error', 'error': result['error']})
    else:
        return jsonify({'status': 'processing'})

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    transcript = session.get('transcript')
    
    if not transcript:
        return jsonify({'error': 'No transcript available'})
    
    task_id = task_queue.add_task(processor.generate_word_cloud, (transcript,))
    
    return jsonify({'task_id': task_id})

@app.route('/wordcloud_result/<task_id>')
def wordcloud_result(task_id):
    result = task_queue.get_result(task_id)
    
    if result['status'] == 'completed':
        return jsonify({'status': 'completed', 'image_data': result['result']})
    elif result['status'] == 'error':
        return jsonify({'status': 'error', 'error': result['error']})
    else:
        return jsonify({'status': 'processing'})

@app.route('/compare_transcripts', methods=['POST'])
def compare_transcripts():
    transcript1 = session.get('transcript')
    
    if not transcript1:
        return jsonify({'error': 'No transcript available'})
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # First transcribe the comparison file
        transcribe_task_id = task_queue.add_task(processor.transcribe_audio, (file_path,))
        
        return jsonify({'task_id': transcribe_task_id, 'stage': 'transcription'})
    
    return jsonify({'error': 'Invalid file'})

@app.route('/comparison_transcription_result/<task_id>')
def comparison_transcription_result(task_id):
    result = task_queue.get_result(task_id)
    
    if result['status'] == 'completed':
        transcript2, message = result['result']
        if transcript2:
            # Now compare the transcripts
            transcript1 = session.get('transcript')
            compare_task_id = task_queue.add_task(
                processor.semantic_similarity, 
                (transcript1[:5000], transcript2[:5000])
            )
            return jsonify({
                'status': 'completed', 
                'next_task_id': compare_task_id,
                'transcript2': transcript2[:500] + '...'  # Send preview
            })
        else:
            return jsonify({'status': 'error', 'error': message})
    elif result['status'] == 'error':
        return jsonify({'status': 'error', 'error': result['error']})
    else:
        return jsonify({'status': 'processing'})

@app.route('/comparison_result/<task_id>')
def comparison_result(task_id):
    result = task_queue.get_result(task_id)
    
    if result['status'] == 'completed':
        similarity = result['result']
        return jsonify({
            'status': 'completed', 
            'similarity': similarity,
            'interpretation': get_similarity_interpretation(similarity)
        })
    elif result['status'] == 'error':
        return jsonify({'status': 'error', 'error': result['error']})
    else:
        return jsonify({'status': 'processing'})

def get_similarity_interpretation(similarity):
    if similarity > 0.8:
        return "The transcripts are very similar in content."
    elif similarity > 0.5:
        return "The transcripts have some similarity in content."
    else:
        return "The transcripts are quite different in content."

@app.route('/chatbot_message', methods=['POST'])
def chatbot_message():
    message = request.json.get('message', '')
    transcript = session.get('transcript', '')
    
    if not message:
        return jsonify({'error': 'No message provided'})
    
    # Simple rule-based responses
    if 'hello' in message.lower() or 'hi' in message.lower():
        return jsonify({'response': 'Hello! How can I help you with your transcript?'})
    elif 'help' in message.lower():
        return jsonify({'response': 'I can help you analyze your transcript. Try asking me to summarize it, answer questions about it, or analyze its sentiment.'})
    elif 'summarize' in message.lower() or 'summary' in message.lower():
        if not transcript:
            return jsonify({'response': 'No transcript available. Please transcribe a file first.'})
        
        task_id = task_queue.add_task(processor.summarization, (transcript,))
        # Wait for a short time to get the result
        time.sleep(2)
        result = task_queue.get_result(task_id)
        
        if result['status'] == 'completed':
            return jsonify({'response': f"Here's a summary of your transcript:\n\n{result['result']}"})
        else:
            return jsonify({'response': "I'm working on summarizing your transcript. This might take a moment."})
    elif 'sentiment' in message.lower() or 'emotion' in message.lower():
        if not transcript:
            return jsonify({'response': 'No transcript available. Please transcribe a file first.'})
        
        return jsonify({'response': "I can analyze the sentiment of your transcript. Use the 'Analyze Sentiment' button in the analysis section."})
    elif 'question' in message.lower() or '?' in message:
        if not transcript:
            return jsonify({'response': 'No transcript available. Please transcribe a file first.'})
        
        # Extract the question - simple heuristic
        if '?' in message:
            question = message.split('?')[0] + '?'
        else:
            question = message
        
        task_id = task_queue.add_task(processor.answer_question, (transcript, question))
        # Wait for a short time to get the result
        time.sleep(2)
        result = task_queue.get_result(task_id)
        
        if result['status'] == 'completed':
            return jsonify({'response': f"Answer: {result['result']}"})
        else:
            return jsonify({'response': "I'm thinking about your question. This might take a moment."})
    else:
        return jsonify({'response': "I'm not sure how to respond to that. Try asking me to summarize your transcript or answer a specific question about it."})

if __name__ == '__main__':
    app.run(debug=True)
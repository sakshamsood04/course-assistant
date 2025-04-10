import os
import tempfile
import time
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from doc_proc import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chatbot import CourseAssistantChatbot

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize components
processor = DocumentProcessor()
vector_store = VectorStoreManager()
chatbot = CourseAssistantChatbot()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get list of available classes
    classes = chatbot.get_available_classes()
    return render_template('index.html', classes=classes)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        class_name = data.get('class_name')
        question = data.get('question')
        
        if not class_name or not question:
            return jsonify({"error": "Class name and question are required"}), 400
        
        # Get conversation history from session or initialize empty list
        conversation_history = session.get('conversation_history', [])
        
        # Generate response
        response = chatbot.generate_response(
            class_name=class_name,
            question=question,
            chat_history=conversation_history
        )
        
        # Add to conversation history
        conversation_history.append((question, response["answer"]))
        
        # Keep only the last 10 exchanges to prevent the context from getting too large
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        # Update session
        session['conversation_history'] = conversation_history
        session['current_class'] = class_name
        
        return jsonify({
            "answer": response["answer"],
            "sources": response["sources"],
            "tokens_used": response["tokens_used"],
            "cost": response["cost"]
        })
    
    # For GET requests
    classes = chatbot.get_available_classes()
    current_class = request.args.get('class_name') or session.get('current_class')
    
    # If a class is specified and not in the list, default to the first class
    if current_class and current_class not in classes and classes:
        current_class = classes[0]
    
    return render_template('chat.html', classes=classes, current_class=current_class)

@app.route('/add-class', methods=['GET', 'POST'])
def add_class():
    if request.method == 'POST':
        class_name = request.form.get('class_name')
        
        if not class_name:
            flash('Please enter a class name.')
            return redirect(url_for('add_class'))
        
        # Check if any files were uploaded
        has_textbook = 'textbook' in request.files and request.files['textbook'].filename
        has_lecture_notes = any(f and f.filename for f in request.files.getlist('lecture_notes'))
        has_assignments = any(f and f.filename for f in request.files.getlist('assignments'))
        
        if not (has_textbook or has_lecture_notes or has_assignments):
            flash('Please upload at least one file.')
            return redirect(url_for('add_class'))
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            textbook_path = None
            lecture_notes_dir = os.path.join(temp_dir, 'lecture_notes')
            assignments_dir = os.path.join(temp_dir, 'assignments')
            
            os.makedirs(lecture_notes_dir, exist_ok=True)
            os.makedirs(assignments_dir, exist_ok=True)
            
            # Process textbook
            if has_textbook:
                textbook_file = request.files['textbook']
                if textbook_file and allowed_file(textbook_file.filename):
                    textbook_path = os.path.join(temp_dir, secure_filename(textbook_file.filename))
                    textbook_file.save(textbook_path)
            
            # Process lecture notes
            for file in request.files.getlist('lecture_notes'):
                if file and file.filename and allowed_file(file.filename):
                    file_path = os.path.join(lecture_notes_dir, secure_filename(file.filename))
                    file.save(file_path)
            
            # Process assignments
            for file in request.files.getlist('assignments'):
                if file and file.filename and allowed_file(file.filename):
                    file_path = os.path.join(assignments_dir, secure_filename(file.filename))
                    file.save(file_path)
            
            # Check if any valid files were saved
            has_valid_files = (
                (textbook_path and os.path.exists(textbook_path)) or
                any(os.path.exists(os.path.join(lecture_notes_dir, f)) for f in os.listdir(lecture_notes_dir)) or
                any(os.path.exists(os.path.join(assignments_dir, f)) for f in os.listdir(assignments_dir))
            )
            
            if not has_valid_files:
                flash('No valid PDF files were uploaded.')
                return redirect(url_for('add_class'))
            
            # Process the class materials
            # Note: This may take some time for large files
            flash('Processing files... This may take a few minutes for large files.')
            
            success = processor.process_class_materials(
                class_name=class_name,
                textbook_path=textbook_path if textbook_path and os.path.exists(textbook_path) else None,
                lecture_notes_dir=lecture_notes_dir if os.listdir(lecture_notes_dir) else None,
                assignments_dir=assignments_dir if os.listdir(assignments_dir) else None
            )
            
            if success:
                flash(f'Successfully added class "{class_name}".')
                # Reset session data for the new class
                session.pop('conversation_history', None)
                session['current_class'] = class_name
                return redirect(url_for('chat', class_name=class_name))
            else:
                flash('Error processing class materials. Please try again.')
                return redirect(url_for('add_class'))
    
    return render_template('add_class.html')

@app.route('/reset-chat', methods=['POST'])
def reset_chat():
    # Clear conversation history
    session.pop('conversation_history', None)
    
    # Get current class
    current_class = session.get('current_class')
    
    return jsonify({"status": "success", "message": "Conversation reset", "class": current_class})

@app.route('/class-info/<class_name>')
def class_info(class_name):
    # Get class info
    info = vector_store.get_class_info(class_name)
    
    if not info.get('exists', False):
        return jsonify({"error": "Class not found"}), 404
    
    return jsonify(info)

@app.route('/list-classes')
def list_classes():
    # Get list of classes
    classes = chatbot.get_available_classes()
    
    # Get info for each class
    class_info = []
    for class_name in classes:
        info = vector_store.get_class_info(class_name)
        if info.get('exists', False):
            class_info.append(info)
    
    return jsonify(class_info)

@app.route('/delete-class/<class_name>', methods=['POST'])
def delete_class(class_name):
    # Check if class exists
    info = vector_store.get_class_info(class_name)
    
    if not info.get('exists', False):
        return jsonify({"error": "Class not found"}), 404
    
    # Delete class
    success = vector_store.delete_class(class_name)
    
    if success:
        # Clear session if it was the current class
        if session.get('current_class') == class_name:
            session.pop('current_class', None)
            session.pop('conversation_history', None)
        
        return jsonify({"status": "success", "message": f"Class '{class_name}' deleted successfully"})
    else:
        return jsonify({"error": "Failed to delete class"}), 500

@app.template_filter('to_date')
def to_date(timestamp):
    """Convert timestamp to formatted date string"""
    import datetime
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
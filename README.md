# CourseTA - AI Teaching Assistant

CourseTA is an AI-powered teaching assistant that creates personalized chatbots for college courses. Unlike generic AI assistants, CourseTA is grounded in your actual course materials, providing accurate, contextually relevant answers with source citations.

## 🚀 Features

### 📚 **Multi-Document Support**
- Upload textbooks, lecture notes, and assignments
- Supports PDF files of any size
- Processes multiple documents per course

### 🧠 **Intelligent RAG System**
- Uses Retrieval-Augmented Generation (RAG) for accurate responses
- Semantic search through course materials
- GPT-4 powered responses grounded in course content

### 🎯 **Source Attribution**
- Every response includes exact citations
- References specific pages and document types
- Maintains academic integrity

### 🗂️ **Multi-Class Management**
- Organize materials by course/class
- Switch between different courses seamlessly
- Independent knowledge bases per class

### 💬 **Interactive Chat Interface**
- Clean, responsive web interface
- Conversation history tracking
- Real-time responses with typing indicators

### 🔐 **Secure Access**
- Password-protected access
- Session management
- File upload validation

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Course-Assistant.git
   cd Course-Assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   
   Open your browser and navigate to `http://localhost:5000`
   
   **Default login password:** `$@k$h@M`

## 📖 Usage Guide

### Adding Your First Course

1. **Login** with the default password
2. **Click "Add Your First Class"**
3. **Fill in course details:**
   - Course name (e.g., "Introduction to Machine Learning")
   - Course description
4. **Upload materials:**
   - **Textbook**: Main course textbook (PDF)
   - **Lecture Notes**: Multiple PDF files from lectures
   - **Assignments**: Additional supplementary materials
5. **Click "Create Class"** - Processing may take a few minutes

### Chatting with Your CourseTA

1. **Navigate to the Chat page**
2. **Select your course** from the dropdown
3. **Ask questions** about your course materials
4. **Get responses** with exact source citations

### Example Questions
- "What are the key concepts from Chapter 3?"
- "Explain the neural network architecture from lecture 5"
- "What did the professor say about backpropagation?"
- "Summarize the main points from the midterm study guide"

## 🔧 How It Works

### 1. **Document Processing**
- PDFs are parsed and split into semantic chunks
- Each chunk is enriched with metadata (source, page, document type)
- Text is converted to embeddings using OpenAI's embedding model

### 2. **Vector Storage**
- Embeddings are stored in ChromaDB for efficient retrieval
- Each course has its own isolated vector database
- Persistent storage ensures data survives restarts

### 3. **RAG Pipeline**
- User questions are embedded and used to search the vector store
- Top 5 most relevant chunks are retrieved
- Context is assembled with source information

### 4. **Response Generation**
- GPT-4 generates responses using only retrieved context
- Responses are grounded in course materials
- Source citations are automatically included

### Railway Deployment
This project is configured for Railway deployment with:
- Persistent volume storage
- Automatic scaling
- Environment variable management

## 🧪 Technology Stack

- **Backend**: Flask, Python 3.8+
- **AI/ML**: OpenAI GPT-4, LangChain
- **Vector Database**: ChromaDB
- **Document Processing**: PyPDF, PyMuPDF
- **Frontend**: HTML, Bootstrap, JavaScript
- **Deployment**: Railway, Gunicorn

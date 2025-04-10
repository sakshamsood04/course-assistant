import os
from typing import List, Any, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the document processor with Railway volume support.
        
        Args:
            openai_api_key: OpenAI API key for embeddings (optional)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def process_pdf(self, pdf_path: str, class_name: str, document_type: str) -> List[Document]:
        """
        Process a single PDF file and return chunked documents with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            class_name: Name of the class
            document_type: Type of document (e.g., 'textbook', 'lecture_notes')
            
        Returns:
            List of LangChain Document objects
        """
        try:
            print(f"Processing {pdf_path}...")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                print(f"No content extracted from {pdf_path}")
                return []
            
            # Add metadata to documents
            filename = os.path.basename(pdf_path)
            
            for doc in documents:
                doc.metadata.update({
                    "source": pdf_path,
                    "filename": filename,
                    "class_name": class_name,
                    "document_type": document_type
                })
            
            # Split documents into chunks
            chunked_documents = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunked_documents)} chunks from {pdf_path}")
            
            return chunked_documents
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return []
    
    def process_directory(self, directory: str, class_name: str, document_type: str) -> List[Document]:
        """
        Process all PDF files in a directory and return chunked documents.
        
        Args:
            directory: Directory containing PDF files
            class_name: Name of the class
            document_type: Type of documents in the directory
            
        Returns:
            List of LangChain Document objects
        """
        try:
            print(f"Processing directory {directory}...")
            
            # Create directory loader for PDF files
            loader = DirectoryLoader(
                directory,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            
            # Load documents
            documents = loader.load()
            
            if not documents:
                print(f"No documents found in {directory}")
                return []
            
            # Add metadata to documents
            for doc in documents:
                filename = os.path.basename(doc.metadata.get("source", "unknown"))
                doc.metadata.update({
                    "filename": filename,
                    "class_name": class_name,
                    "document_type": document_type
                })
            
            # Split documents into chunks
            chunked_documents = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunked_documents)} chunks from {directory}")
            
            return chunked_documents
            
        except Exception as e:
            print(f"Error processing directory {directory}: {e}")
            return []
    
    def create_vector_store(self, documents: List[Document], class_name: str) -> Any:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of LangChain Document objects
            class_name: Name of the class
            
        Returns:
            Chroma vector store or None if failed
        """
        if not documents:
            print("No documents to create vector store from")
            return None
        
        try:
            # Create a sanitized collection name
            collection_name = class_name.replace(" ", "_").lower()
            
            # Check for Railway volume mount path
            railway_volume_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
            
            if railway_volume_path and os.path.exists(railway_volume_path):
                # Use Railway volume for persistence
                base_persist_directory = os.path.join(railway_volume_path, "chroma_db")
                print(f"Using Railway volume for persistence at: {base_persist_directory}")
            else:
                # Fallback to local directory
                base_persist_directory = "chroma_db"
                print(f"Using local directory for persistence at: {base_persist_directory}")
            
            # Create full path for this collection
            collection_path = os.path.join(base_persist_directory, collection_name)
            os.makedirs(collection_path, exist_ok=True)
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=collection_path,
                collection_name=collection_name
            )
            
            # Explicitly persist the vector store
            try:
                if hasattr(vector_store, 'persist'):
                    vector_store.persist()
                    print(f"Vector store persisted at {collection_path}")
            except Exception as e:
                print(f"Warning: Could not persist vector store, but it should still be usable: {e}")
            
            print(f"Created vector store for class '{class_name}' with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    
    def process_class_materials(
        self, 
        class_name: str, 
        textbook_path: Optional[str] = None,
        lecture_notes_dir: Optional[str] = None,
        assignments_dir: Optional[str] = None
    ) -> bool:
        """
        Process all materials for a class and create a vector store.
        
        Args:
            class_name: Name of the class
            textbook_path: Path to the textbook PDF (optional)
            lecture_notes_dir: Directory containing lecture notes PDFs (optional)
            assignments_dir: Directory containing assignment PDFs (optional)
            
        Returns:
            True if successful, False otherwise
        """
        all_documents = []
        
        # Process textbook if provided
        if textbook_path and os.path.isfile(textbook_path) and textbook_path.lower().endswith('.pdf'):
            textbook_docs = self.process_pdf(textbook_path, class_name, "textbook")
            all_documents.extend(textbook_docs)
        
        # Process lecture notes if provided
        if lecture_notes_dir and os.path.isdir(lecture_notes_dir):
            lecture_docs = self.process_directory(lecture_notes_dir, class_name, "lecture_notes")
            all_documents.extend(lecture_docs)
        
        # Process assignments if provided
        if assignments_dir and os.path.isdir(assignments_dir):
            assignment_docs = self.process_directory(assignments_dir, class_name, "assignments")
            all_documents.extend(assignment_docs)
        
        if not all_documents:
            print("No documents were processed successfully")
            return False
        
        # Create vector store
        vector_store = self.create_vector_store(all_documents, class_name)
        
        return vector_store is not None
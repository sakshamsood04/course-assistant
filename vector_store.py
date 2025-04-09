import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Import the correct embeddings class
from langchain_openai import OpenAIEmbeddings

# Import the correct Chroma class
try:
    from langchain_chroma import Chroma
    print("v1.1")
except ImportError:
    from langchain.vectorstores import Chroma
    print("v1.2")

from langchain_core.documents import Document

# Load environment variables
load_dotenv()

class VectorStoreManager:
    def __init__(self, base_persist_directory: str = "chroma_db", openai_api_key: Optional[str] = None):
        """
        Initialize the vector store manager.
        
        Args:
            base_persist_directory: Base directory for all vector stores
            openai_api_key: OpenAI API key for embeddings
        """
        self.base_persist_directory = base_persist_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Create base directory if it doesn't exist
        os.makedirs(base_persist_directory, exist_ok=True)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
    
    def get_collection_path(self, class_name: str) -> str:
        """Get the path to a collection directory."""
        collection_name = class_name.replace(" ", "_").lower()
        return os.path.join(self.base_persist_directory, collection_name)
    
    def get_vector_store(self, class_name: str) -> Optional[Chroma]:
        """
        Get a vector store for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Chroma vector store or None if not found
        """
        collection_name = class_name.replace(" ", "_").lower()
        collection_path = self.get_collection_path(class_name)
        
        # Check if collection directory exists
        if not os.path.exists(collection_path):
            # Try to find a case-insensitive match
            available_collections = self.list_available_classes()
            for available_class in available_collections:
                if available_class.lower() == class_name.lower():
                    # If matched, use the correct case version
                    collection_name = available_class.replace(" ", "_").lower()
                    collection_path = self.get_collection_path(available_class)
                    break
            
            # If still not found
            if not os.path.exists(collection_path):
                print(f"Collection for class '{class_name}' does not exist")
                return None
        
        try:
            # Load vector store
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=collection_path
            )
            
            return vector_store
        except Exception as e:
            print(f"Error loading vector store for class '{class_name}': {e}")
            return None
    
    def query_vector_store(
        self, 
        class_name: str, 
        query_text: str, 
        n_results: int = 5,
        include_metadata: bool = True
    ) -> Tuple[List[Document], List[float]]:
        """
        Query a vector store.
        
        Args:
            class_name: Name of the class
            query_text: Query text
            n_results: Number of results to return
            include_metadata: Whether to include document metadata in results
            
        Returns:
            Tuple of (documents, similarities) or ([], []) if error
        """
        vector_store = self.get_vector_store(class_name)
        
        if not vector_store:
            return [], []
        
        try:
            # Query vector store
            documents_with_scores = vector_store.similarity_search_with_relevance_scores(
                query=query_text,
                k=n_results
            )
            
            # Extract documents and scores
            documents = [doc for doc, _ in documents_with_scores]
            scores = [score for _, score in documents_with_scores]
            
            return documents, scores
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return [], []
    
    def list_available_classes(self) -> List[str]:
        """
        List all available classes in the database.
        
        Returns:
            List of class names
        """
        try:
            # Check if base directory exists
            if not os.path.exists(self.base_persist_directory):
                return []
                
            # Get all subdirectories in the base directory
            subdirs = [d for d in os.listdir(self.base_persist_directory) 
                      if os.path.isdir(os.path.join(self.base_persist_directory, d))]
            
            # Filter out directories that don't look like valid Chroma collections
            valid_subdirs = []
            for subdir in subdirs:
                path = os.path.join(self.base_persist_directory, subdir)
                # Check if it contains chroma files or a chroma.sqlite3 file
                if (os.path.exists(os.path.join(path, "chroma.sqlite3")) or 
                    os.path.exists(os.path.join(path, "index"))):
                    valid_subdirs.append(subdir)
            
            # Convert collection names back to class names
            class_names = [subdir.replace("_", " ").title() for subdir in valid_subdirs]
            
            return class_names
        except Exception as e:
            print(f"Error listing classes: {e}")
            return []
    
    def get_class_info(self, class_name: str) -> Dict[str, Any]:
        """
        Get information about a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Dictionary with class information
        """
        vector_store = self.get_vector_store(class_name)
        
        if not vector_store:
            return {"exists": False}
        
        try:
            # Get collection info
            collection = vector_store._collection
            count = collection.count()
            
            # Get document types
            document_types = set()
            
            # Sample a few documents to get metadata
            if count > 0:
                ids = collection.get(limit=min(10, count))["ids"]
                metadatas = collection.get(ids=ids)["metadatas"]
                
                for metadata in metadatas:
                    if metadata and "document_type" in metadata:
                        document_types.add(metadata["document_type"])
            
            return {
                "exists": True,
                "class_name": class_name,
                "document_count": count,
                "document_types": list(document_types)
            }
        except Exception as e:
            print(f"Error getting class info: {e}")
            return {"exists": False, "error": str(e)}
    
    def delete_class(self, class_name: str) -> bool:
        """
        Delete a class from the database.
        
        Args:
            class_name: Name of the class
            
        Returns:
            True if successful, False otherwise
        """
        collection_path = self.get_collection_path(class_name)
        
        # Check if collection directory exists
        if not os.path.exists(collection_path):
            print(f"Collection for class '{class_name}' does not exist")
            return False
        
        try:
            # Get vector store
            vector_store = self.get_vector_store(class_name)
            
            if vector_store:
                # Delete collection
                vector_store._collection.delete(ignore_missing=True)
            
            # Remove directory
            import shutil
            shutil.rmtree(collection_path)
            
            print(f"Deleted class '{class_name}'")
            return True
        except Exception as e:
            print(f"Error deleting class '{class_name}': {e}")
            return False

# Example usage
if __name__ == "__main__":
    manager = VectorStoreManager()
    
    # List available classes
    classes = manager.list_available_classes()
    print(f"Available classes: {classes}")
    
    # If there are classes, query one
    if classes:
        class_name = classes[0]
        query = "Explain gradient descent algorithm"
        
        print(f"\nQuerying class '{class_name}' with: '{query}'")
        docs, scores = manager.query_vector_store(class_name, query)
        
        print(f"Found {len(docs)} results")
        for i, (doc, score) in enumerate(zip(docs, scores)):
            print(f"\nResult {i+1} (score: {score:.4f}):")
            print(f"Source: {doc.metadata.get('filename', 'unknown')}")
            print(f"Type: {doc.metadata.get('document_type', 'unknown')}")
            print(f"Content preview: {doc.page_content[:150]}...")
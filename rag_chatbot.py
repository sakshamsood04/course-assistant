import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Import the correct Chroma class
# try:
from langchain_chroma import Chroma
#     print("ragchatbot 1.1")
# except ImportError:
#     from langchain.vectorstores import Chroma
#     print("ragchatbot 1.2")

from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the correct callback tracker
# try:
#     from langchain_core.callbacks import get_openai_callback
#     print("ragchatbot 2.1")
# except ImportError:
#     try:
from langchain.callbacks import get_openai_callback
    #     print("ragchatbot 2.2")
    # except ImportError:
    #     # Create a simple callback tracker if neither import works
    #     print("ragchatbot 2.3-created its own open ai callback handler")
    #     class OpenAICallbackHandler:
    #         def __init__(self):
    #             self.total_tokens = 0
    #             self.prompt_tokens = 0
    #             self.completion_tokens = 0
    #             self.total_cost = 0.0
                
    #         def __enter__(self):
    #             return self
                
    #         def __exit__(self, exc_type, exc_value, traceback):
    #             pass
        
    #     def get_openai_callback():
    #         return OpenAICallbackHandler()

from langchain.memory import ConversationBufferMemory
from vector_store import VectorStoreManager

# Load environment variables
load_dotenv()

class CourseAssistantChatbot:
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        vector_store_directory: str = "chroma_db"
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            openai_api_key: OpenAI API key
            model_name: OpenAI model name to use
            temperature: Temperature for model generation (0-1)
            vector_store_directory: Directory for vector stores
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager(
            base_persist_directory=vector_store_directory,
            openai_api_key=self.openai_api_key
        )
        
        # Define system prompt template
        self.system_template = """
        You are CourseTA, a helpful and knowledgeable teaching assistant for the course: {class_name}.
        
        Your goal is to help students understand concepts, answer their questions, and provide guidance based on the course materials.
        
        When answering:
        1. Only use information from the provided context - do not use external knowledge
        2. If the answer isn't in the context, say "I don't have enough information to answer that question based on the course materials."
        3. Cite specific lectures, textbook sections, or other materials when possible (mention the source filename)
        4. Explain concepts clearly as if teaching a student
        5. Use appropriate formatting (bullet points, italics, etc.) to make your answer easy to understand
        
        Context information:
        {context}
        
        Remember: You are a teaching assistant, so maintain a helpful, educational tone. 
        Be concise but thorough, and make sure your explanations are clear and accessible.
        """
    
    def get_available_classes(self) -> List[str]:
        """Get a list of available classes."""
        return self.vector_store_manager.list_available_classes()
    
    def generate_response(
        self, 
        class_name: str, 
        question: str, 
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a question.
        
        Args:
            class_name: Name of the class
            question: User's question
            chat_history: List of (question, answer) tuples from previous conversation
            
        Returns:
            Dictionary with response and metadata
        """
        # Get info about the class
        class_info = self.vector_store_manager.get_class_info(class_name)
        
        if not class_info.get("exists", False):
            return {
                "answer": f"Sorry, I couldn't find any information for the class '{class_name}'. Please make sure the class exists and has been properly added to the system.",
                "sources": [],
                "tokens_used": 0,
                "cost": 0.0
            }
        
        # Get vector store
        vector_store = self.vector_store_manager.get_vector_store(class_name)
        
        if not vector_store:
            return {
                "answer": f"Error: Could not load vector store for class '{class_name}'.",
                "sources": [],
                "tokens_used": 0,
                "cost": 0.0
            }
        
        # Format chat history for the context
        formatted_history = []
        if chat_history:
            for user_msg, ai_msg in chat_history:
                formatted_history.append(HumanMessage(content=user_msg))
                formatted_history.append(AIMessage(content=ai_msg))
        
        try:
            # Track token usage and cost
            with get_openai_callback() as cb:
                # Create prompt
                prompt = ChatPromptTemplate.from_template(self.system_template)
                
                # Direct retrieval to get context first
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                
                # Get relevant documents
                retrieved_docs = retriever.invoke(question)
                
                # Format context from retrieved documents
                contexts = []
                for i, doc in enumerate(retrieved_docs):
                    source_info = f"Source: {doc.metadata.get('document_type', 'Unknown')} - {doc.metadata.get('filename', 'Unknown')}"
                    if 'page' in doc.metadata:
                        source_info += f", Page {doc.metadata['page']}"
                    
                    contexts.append(f"[Document {i+1}] {source_info}\n{doc.page_content}\n")
                
                context_text = "\n\n".join(contexts)
                
                # Generate the response
                response = self.llm.invoke(
                    prompt.format(
                        class_name=class_name,
                        context=context_text
                    )
                )
                
                # Format sources
                sources = []
                for doc in retrieved_docs:
                    source = {
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "document_type": doc.metadata.get("document_type", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    }
                    sources.append(source)
                
                return {
                    "answer": response.content,
                    "sources": sources,
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost
                }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Sorry, I encountered an error while generating a response: {str(e)}",
                "sources": [],
                "tokens_used": 0,
                "cost": 0.0
            }
    
    def reset_conversation(self, class_name: str) -> bool:
        """
        Reset the conversation history for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            True if successful, False otherwise
        """
        # Since we're not storing conversation state in this implementation, 
        # this method doesn't need to do anything special
        return True

# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = CourseAssistantChatbot()
    
    # List available classes
    classes = chatbot.get_available_classes()
    
    if not classes:
        print("No classes available. Please add a class first.")
    else:
        print("Available classes:")
        for i, class_name in enumerate(classes, 1):
            print(f"{i}. {class_name}")
        
        # Select a class
        class_idx = int(input("\nSelect a class (enter number): ")) - 1
        
        if 0 <= class_idx < len(classes):
            selected_class = classes[class_idx]
            
            print(f"\nYou are now chatting with CourseTA for '{selected_class}'")
            print("Type 'exit' to end the conversation\n")
            
            chat_history = []
            
            while True:
                # Get user question
                question = input("You: ")
                
                if question.lower() in ["exit", "quit", "bye"]:
                    print("CourseTA: Goodbye! Feel free to chat again if you have more questions.")
                    break
                
                # Generate response
                response = chatbot.generate_response(selected_class, question, chat_history)
                
                # Print answer
                print(f"\nCourseTA: {response['answer']}\n")
                
                # Add to chat history
                chat_history.append((question, response["answer"]))
                
                # Print source info
                if response["sources"]:
                    print("Sources:")
                    for i, source in enumerate(response["sources"], 1):
                        print(f"  {i}. {source['document_type']}: {source['filename']}, Page: {source['page']}")
                    print()
                
                # Print token usage
                print(f"Tokens used: {response['tokens_used']}, Cost: ${response['cost']:.6f}\n")
        else:
            print("Invalid selection.")
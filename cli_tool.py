import argparse
from doc_proc import DocumentProcessor
from vector_store import VectorStoreManager

def main():
    parser = argparse.ArgumentParser(description='Teaching Assistant Chatbot - Document Processing CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add a new class
    add_class_parser = subparsers.add_parser('add-class', help='Add a new class')
    add_class_parser.add_argument('--name', required=True, help='Name of the class')
    add_class_parser.add_argument('--textbook', help='Path to the textbook PDF')
    add_class_parser.add_argument('--lecture-notes', help='Path to a directory containing lecture notes PDFs')
    add_class_parser.add_argument('--assignments', help='Path to a directory containing assignment PDFs')
    
    # List available classes
    list_parser = subparsers.add_parser('list-classes', help='List all available classes')
    
    # Get class info
    info_parser = subparsers.add_parser('class-info', help='Get information about a class')
    info_parser.add_argument('--name', required=True, help='Name of the class')
    
    # Query a class
    query_parser = subparsers.add_parser('query', help='Query a class')
    query_parser.add_argument('--name', required=True, help='Name of the class')
    query_parser.add_argument('--question', required=True, help='Question to ask')
    query_parser.add_argument('--results', type=int, default=5, help='Number of results to return')
    
    # Delete a class
    delete_parser = subparsers.add_parser('delete-class', help='Delete a class')
    delete_parser.add_argument('--name', required=True, help='Name of the class')
    
    args = parser.parse_args()
    
    if args.command == 'add-class':
        add_class(args)
    elif args.command == 'list-classes':
        list_classes()
    elif args.command == 'class-info':
        get_class_info(args)
    elif args.command == 'query':
        query_class(args)
    elif args.command == 'delete-class':
        delete_class(args)
    else:
        parser.print_help()

def add_class(args):
    """Add a new class with materials."""
    processor = DocumentProcessor()
    
    print(f"Processing materials for class '{args.name}'...")
    
    success = processor.process_class_materials(
        class_name=args.name,
        textbook_path=args.textbook,
        lecture_notes_dir=args.lecture_notes,
        assignments_dir=args.assignments
    )
    
    if success:
        print(f"Successfully processed materials for class '{args.name}'")
    else:
        print(f"Failed to process materials for class '{args.name}'")

def list_classes():
    """List all available classes."""
    manager = VectorStoreManager()
    classes = manager.list_available_classes()
    
    if not classes:
        print("No classes found")
        return
    
    print("Available classes:")
    for i, class_name in enumerate(classes, 1):
        # Get class info
        info = manager.get_class_info(class_name)
        doc_count = info.get('document_count', 'unknown')
        doc_types = info.get('document_types', [])
        
        doc_types_str = ", ".join(doc_types) if doc_types else "none"
        
        print(f"{i}. {class_name} ({doc_count} documents, types: {doc_types_str})")

def get_class_info(args):
    """Get information about a class."""
    manager = VectorStoreManager()
    info = manager.get_class_info(args.name)
    
    if not info.get('exists', False):
        print(f"Class '{args.name}' not found")
        return
    
    print(f"Class: {info['class_name']}")
    print(f"Number of documents: {info['document_count']}")
    print(f"Document types: {', '.join(info['document_types'])}")

def query_class(args):
    """Query a class."""
    manager = VectorStoreManager()
    
    print(f"Querying class '{args.name}' with question: '{args.question}'")
    
    documents, scores = manager.query_vector_store(
        class_name=args.name,
        query_text=args.question,
        n_results=args.results
    )
    
    if not documents:
        print(f"No results found or class '{args.name}' does not exist")
        return
    
    print(f"\nFound {len(documents)} results:")
    
    for i, (doc, score) in enumerate(zip(documents, scores), 1):
        print(f"\nResult {i} (relevance: {score:.4f}):")
        print(f"Source: {doc.metadata.get('filename', 'unknown')}")
        print(f"Document type: {doc.metadata.get('document_type', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', 'unknown')}")
        
        # Display content preview (up to 200 chars)
        content_preview = doc.page_content[:200]
        if len(doc.page_content) > 200:
            content_preview += "..."
        
        print(f"Content: {content_preview}")

def delete_class(args):
    """Delete a class."""
    manager = VectorStoreManager()
    
    # Ask for confirmation
    confirm = input(f"Are you sure you want to delete class '{args.name}'? This cannot be undone! (y/n): ")
    
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return
    
    success = manager.delete_class(args.name)
    
    if success:
        print(f"Successfully deleted class '{args.name}'")
    else:
        print(f"Failed to delete class '{args.name}'")

if __name__ == "__main__":
    main()
import os
import argparse
from dotenv import load_dotenv
import json
from rag_chatbot import CourseAssistantChatbot

# Load environment variables
load_dotenv()

def format_sources(sources):
    """Format source information for display"""
    if not sources:
        return "No specific sources used"
    
    source_info = []
    for i, source in enumerate(sources, 1):
        source_info.append(f"{i}. {source['document_type']}: {source['filename']}, Page: {source['page']}")
    
    return "\n".join(source_info)

def interactive_mode(class_name, model_name, temperature):
    """Run chatbot in interactive mode"""
    # Initialize chatbot
    chatbot = CourseAssistantChatbot(
        model_name=model_name,
        temperature=temperature
    )
    
    # Check if class exists - case insensitive matching
    available_classes = chatbot.get_available_classes()
    matching_class = None
    
    for cls in available_classes:
        if cls.lower() == class_name.lower():
            matching_class = cls
            break
    
    if not matching_class:
        print(f"Class '{class_name}' not found. Available classes:")
        for i, cls in enumerate(available_classes, 1):
            print(f"{i}. {cls}")
        
        if available_classes:
            choice = input("\nSelect a class by number or enter 'q' to quit: ")
            if choice.lower() == 'q':
                return
            try:
                class_idx = int(choice) - 1
                if 0 <= class_idx < len(available_classes):
                    class_name = available_classes[class_idx]
                else:
                    print("Invalid selection. Exiting.")
                    return
            except:
                print("Invalid input. Exiting.")
                return
        else:
            print("No classes available. Please add a class first.")
            return
    else:
        class_name = matching_class  # Use the correctly-cased class name
    
    # Start conversation
    print(f"\n{'='*50}")
    print(f"Starting conversation with CourseTA for '{class_name}'")
    print(f"Model: {model_name}, Temperature: {temperature}")
    print("Type 'exit' to end the conversation or 'reset' to reset chat history")
    print(f"{'='*50}\n")
    
    chat_history = []
    
    while True:
        # Get user question
        question = input("\nYou: ")
        
        if question.lower() in ["exit", "quit", "bye"]:
            print("\nCourseTA: Goodbye! Feel free to chat again if you have more questions.")
            break
        
        if question.lower() == "reset":
            chat_history = []
            print("\nChat history has been reset.")
            continue
        
        # Generate response
        print("\nThinking...")
        response = chatbot.generate_response(class_name, question, chat_history)
        
        # Print answer
        print(f"\nCourseTA: {response['answer']}")
        
        # Add to chat history
        chat_history.append((question, response["answer"]))
        
        # Print source info
        if response["sources"]:
            print("\nSources:")
            print(format_sources(response["sources"]))
        
        # Print token usage
        print(f"\nTokens used: {response['tokens_used']}, Cost: ${response['cost']:.6f}")

def single_question_mode(class_name, question, model_name, temperature, output_file=None):
    """Run chatbot for a single question"""
    # Initialize chatbot
    chatbot = CourseAssistantChatbot(
        model_name=model_name,
        temperature=temperature
    )
    
    # Check if class exists - case insensitive matching
    available_classes = chatbot.get_available_classes()
    matching_class = None
    
    for cls in available_classes:
        if cls.lower() == class_name.lower():
            matching_class = cls
            break
    
    if not matching_class:
        print(f"Error: Class '{class_name}' not found.")
        print("Available classes:")
        for i, cls in enumerate(available_classes, 1):
            print(f"{i}. {cls}")
        return
    
    # Use the correctly-cased class name
    class_name = matching_class
    
    # Generate response
    print(f"Querying class '{class_name}' with: '{question}'")
    print("Thinking...")
    
    response = chatbot.generate_response(class_name, question)
    
    # Print answer
    print(f"\nCourseTA: {response['answer']}")
    
    # Print source info
    if response["sources"]:
        print("\nSources:")
        print(format_sources(response["sources"]))
    
    # Print token usage
    print(f"\nTokens used: {response['tokens_used']}, Cost: ${response['cost']:.6f}")
    
    # Save to output file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(response, f, indent=2)
        print(f"\nResponse saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='CourseTA - Teaching Assistant Chatbot')
    
    # Add arguments
    parser.add_argument('--class', dest='class_name', help='Name of the class')
    parser.add_argument('--question', help='Question to ask (if not provided, interactive mode is used)')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for model generation (0-1, default: 0.2)')
    parser.add_argument('--output', help='Output file to save response (JSON format)')
    parser.add_argument('--list-classes', action='store_true', help='List available classes')
    
    args = parser.parse_args()
    
    # List classes if requested
    if args.list_classes:
        chatbot = CourseAssistantChatbot()
        classes = chatbot.get_available_classes()
        
        if not classes:
            print("No classes available. Please add a class first.")
        else:
            print("Available classes:")
            for i, cls in enumerate(classes, 1):
                print(f"{i}. {cls}")
        return
    
    # Initialize chatbot
    chatbot = CourseAssistantChatbot()
    classes = chatbot.get_available_classes()
    
    if not classes:
        print("No classes available. Please add a class first.")
        return
    
    # Check if class name is provided
    if not args.class_name:
        print("Available classes:")
        for i, cls in enumerate(classes, 1):
            print(f"{i}. {cls}")
        
        choice = input("\nSelect a class by number: ")
        try:
            class_idx = int(choice) - 1
            if 0 <= class_idx < len(classes):
                args.class_name = classes[class_idx]
            else:
                print("Invalid selection. Exiting.")
                return
        except:
            print("Invalid input. Exiting.")
            return
    else:
        # Check if the provided class name exists in the available classes
        matching_class = None
        for cls in classes:
            if cls.lower() == args.class_name.lower():
                matching_class = cls
                break
        
        if matching_class:
            args.class_name = matching_class
        else:
            print(f"Class '{args.class_name}' not found. Available classes:")
            for i, cls in enumerate(classes, 1):
                print(f"{i}. {cls}")
            
            choice = input("\nSelect a class by number or enter 'q' to quit: ")
            if choice.lower() == 'q':
                return
            try:
                class_idx = int(choice) - 1
                if 0 <= class_idx < len(classes):
                    args.class_name = classes[class_idx]
                else:
                    print("Invalid selection. Exiting.")
                    return
            except:
                print("Invalid input. Exiting.")
                return
    
    # Run in appropriate mode
    if args.question:
        # Single question mode
        single_question_mode(args.class_name, args.question, args.model, args.temperature, args.output)
    else:
        # Interactive mode
        interactive_mode(args.class_name, args.model, args.temperature)

if __name__ == "__main__":
    main()
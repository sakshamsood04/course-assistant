import os
import sys

def adapt_for_railway():
    """
    Adapts the application configuration for Railway deployment.
    
    Railway uses ephemeral filesystem, so we need to ensure:
    1. All necessary directories are created on startup
    2. Environment variables are properly set
    """
    print("Adapting application for Railway deployment...")
    
    # Create necessary directories
    dirs_to_create = [
        'uploads',
        'chroma_db',
        'templates',
        'static',
        'static/css',
        'static/js',
        'static/img'
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check if templates and static exist - they should be in the git repo
    # but we'll verify just in case
    if not os.path.exists('templates'):
        print("WARNING: templates directory not found!")
        print("Make sure your HTML templates are included in your git repository.")
    
    if not os.path.exists('static'):
        print("WARNING: static directory not found!")
        print("Make sure your static files are included in your git repository.")
    
    # Verify environment variables
    required_vars = ['OPENAI_API_KEY', 'FLASK_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your Railway project settings.")
        sys.exit(1)
    
    print("Application successfully adapted for Railway!")

if __name__ == "__main__":
    adapt_for_railway()
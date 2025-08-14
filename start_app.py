#!/usr/bin/env python3
import sys
import subprocess
import time
import signal
import threading
import os
from pathlib import Path

# Disable various logging before starting anything
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["HTTPX_LOG_LEVEL"] = "ERROR"
os.environ["URLLIB3_DISABLE_WARNINGS"] = "1"
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Configure logging
import logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("streamlit").setLevel(logging.ERROR)

def start_backend():
    backend_dir = Path(__file__).parent / "backend"
    
    print("Starting backend server...")
    
    try:
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "error",
            "--no-access-log"
        ], cwd=backend_dir)
        return backend_process
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None

def start_frontend():
    frontend_file = Path(__file__).parent / "frontend" / "main.py"
    
    print("Starting frontend...")
    
    try:
        frontend_process = subprocess.Popen([
            "streamlit", "run", str(frontend_file),
            "--server.port=8501",
            "--server.headless=false", 
            "--browser.gatherUsageStats=false",
            "--logger.level=error"
        ])
        return frontend_process
    except FileNotFoundError:
        print("Error: Streamlit not found. Install with: pip install streamlit")
        return None
    except Exception as e:
        print(f"Error starting frontend: {e}")
        return None

def check_backend_health():
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    backend_process = None
    frontend_process = None
    
    def signal_handler(signum, frame):
        print("\nShutting down...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        backend_process = start_backend()
        if not backend_process:
            return 1
        
        for i in range(30):
            if check_backend_health():
                print("Backend is ready!")
                break
            time.sleep(1)
        else:
            print("Backend failed to start within 30 seconds")
            if backend_process:
                backend_process.terminate()
            return 1
        
        frontend_process = start_frontend()
        if not frontend_process:
            if backend_process:
                backend_process.terminate()
            return 1
        
        print("\nApplication started successfully!")
        print("Backend: http://localhost:8000")
        print("Frontend: http://localhost:8501")
        print("\nPress Ctrl+C to stop")
        
        while True:
            if backend_process.poll() is not None:
                print("Backend process stopped")
                break
            if frontend_process.poll() is not None:
                print("Frontend process stopped") 
                break
            time.sleep(1)
        
    except KeyboardInterrupt:
        pass
    finally:
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
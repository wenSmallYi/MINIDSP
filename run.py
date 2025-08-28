#!/usr/bin/env python3
"""
UMIK-1 Project Launcher Script

This script provides a centralized way to run all UMIK-1 recording programs
with the correct project structure and paths.
"""

import os
import sys
import subprocess
import argparse

# Get project root directory (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

def run_cli_recorder():
    """Run the CLI recorder"""
    os.chdir(PROJECT_ROOT)  # Set working directory to project root
    script_path = os.path.join(SRC_DIR, "umik1_recorder_cli.py")
    subprocess.run([sys.executable, script_path] + sys.argv[2:])

def run_server():
    """Run the FastAPI server"""
    os.chdir(PROJECT_ROOT)  # Set working directory to project root
    script_path = os.path.join(SRC_DIR, "umik1_server.py")
    subprocess.run([sys.executable, script_path])

def main():
    parser = argparse.ArgumentParser(description="UMIK-1 Project Launcher")
    parser.add_argument("mode", choices=["cli", "server"], 
                       help="Run mode: 'cli' for command line interface, 'server' for web API")
    
    if len(sys.argv) < 2:
        parser.print_help()
        return
    
    mode = sys.argv[1]
    
    print(f"Starting UMIK-1 {mode.upper()} mode...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Working directory: {os.getcwd()}")
    
    if mode == "cli":
        run_cli_recorder()
    elif mode == "server":
        run_server()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Setup script for commodity prediction project.
This script automates the environment setup and data download process.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description, check=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def setup_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return
    
    # Create virtual environment
    run_command(
        "python3 -m venv .venv",
        "Creating virtual environment"
    )


def get_activation_command():
    """Get the appropriate activation command based on the OS."""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"


def install_kaggle():
    """Install kaggle package in the virtual environment."""
    if platform.system() == "Windows":
        pip_cmd = ".venv\\Scripts\\pip"
    else:
        pip_cmd = ".venv/bin/pip"
    
    # Upgrade pip first
    run_command(
        f"{pip_cmd} install --upgrade pip",
        "Upgrading pip"
    )
    
    # Install kaggle
    run_command(
        f"{pip_cmd} install kaggle",
        "Installing kaggle package"
    )


def download_competition_data():
    """Download competition data using kaggle CLI."""
    if platform.system() == "Windows":
        kaggle_cmd = ".venv\\Scripts\\kaggle"
    else:
        kaggle_cmd = ".venv/bin/kaggle"
    
    # Check if kaggle credentials exist
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_key = kaggle_dir / "kaggle.json"
    
    if not kaggle_key.exists():
        print("⚠️  Warning: Kaggle credentials not found!")
        print("Please ensure you have:")
        print("1. Downloaded kaggle.json from your Kaggle account")
        print("2. Placed it in ~/.kaggle/kaggle.json")
        print("3. Set proper permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("\nContinuing anyway...")
    
    # Create kaggle directory if it doesn't exist
    kaggle_data_dir = Path("kaggle")
    kaggle_data_dir.mkdir(exist_ok=True)
    
    # Download competition data
    run_command(
        f"{kaggle_cmd} competitions download -c mitsui-commodity-prediction-challenge -p kaggle/",
        "Downloading competition data"
    )
    
    # Unzip the downloaded file
    zip_file = kaggle_data_dir / "mitsui-commodity-prediction-challenge.zip"
    if zip_file.exists():
        run_command(
            f"cd kaggle && unzip mitsui-commodity-prediction-challenge.zip",
            "Extracting competition data"
        )
        print("✅ Competition data extracted successfully")
    else:
        print("⚠️  Warning: Downloaded zip file not found, skipping extraction")


def main():
    """Main setup function."""
    print("🚀 Starting commodity prediction project setup...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("setup.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Create virtual environment
    setup_virtual_environment()
    
    # Step 2: Install kaggle
    install_kaggle()
    
    # Step 3: Download competition data
    download_competition_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print(f"   {get_activation_command()}")
    print("2. Install additional dependencies as needed")
    print("3. Start working on your project!")
    
    # Show activation command for current shell
    print(f"\n💡 To activate the environment in your current shell, run:")
    print(f"   {get_activation_command()}")


if __name__ == "__main__":
    main()

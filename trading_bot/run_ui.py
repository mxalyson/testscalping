import subprocess
import sys

print("Iniciando dashboard...")
subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/dashboard.py"])
"""
=============================================================================
  FAKE NEWS DETECTION — STARTUP / SETUP HELPER
  File: utils/setup.py
  Description:
      Checks whether the trained model exists. If not, auto-trains it.
      Called by app.py on startup so fresh deployments work out of the box.
=============================================================================
"""

import os
import sys

def ensure_model_exists():
    """
    Checks if model/best_model.pkl exists.
    If not, runs model_training.py to create it.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model.pkl')

    if not os.path.exists(model_path):
        print("[SETUP] Model not found. Training now...")
        import subprocess
        result = subprocess.run(
            [sys.executable, 'model_training.py'],
            capture_output=False
        )
        if result.returncode != 0:
            print("[SETUP] Training failed. Please run model_training.py manually.")
            return False
        print("[SETUP] ✅ Training complete.")

    return True

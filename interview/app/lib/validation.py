import logging
import os
import pandas as pd
import numpy as np
from typing import Callable, Any, List

# --- Start: Validation Logging Setup ---
validation_logger = logging.getLogger('analytics_validator')
validation_logger.setLevel(logging.INFO)
validation_logger.propagate = False
if not validation_logger.hasHandlers():
    validation_handler = logging.FileHandler('/tmp/output_validation.log', mode='w')
    validation_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    validation_handler.setFormatter(validation_formatter)
    validation_logger.addHandler(validation_handler)
# --- End: Validation Logging Setup ---

class AnalyticsValidator:
    """
    Validates the INPUTS for each module in the interview analytics pipeline.
    Maintains a DataFrame report of the input validation status.
    """
    def __init__(self):
        self.logger = validation_logger
        self.modules = [
            'Face Extraction', 'Audio Features', 'FER',
            'Valence Arousal', 'FACS', 'Behavioral Features', 'Audio Analysis'
        ]
        self.report_df = pd.DataFrame({
            'Module': self.modules,
            'Input Status': 'Pending',
            'Details': ''
        })

    def _update_report(self, module_name: str, status: str, details: str = "", api: Any = None):
        """Updates the internal DataFrame and optionally logs to the API."""
        if module_name not in self.report_df['Module'].values:
            self.logger.error(f"Module '{module_name}' not found in validation report.")
            return
        idx = self.report_df.index[self.report_df['Module'] == module_name][0]
        self.report_df.loc[idx, 'Input Status'] = status
        self.report_df.loc[idx, 'Details'] = details
        
        log_message = f"Input Validation: [{module_name}] - {status}. {details}".strip()
        self.logger.info(log_message)
        if api:
            api.log(log_message)

    def validate_inputs(self, module_name: str, api: Any, *args) -> bool:
        """
        Validates inputs for a given module, updates the report, and returns boolean validity.
        """
        try:
            validation_method_name = f"_validate_inputs_for_{module_name.lower().replace(' ', '_')}"
            validation_method = getattr(self, validation_method_name)

            is_valid, details = validation_method(*args)

            if is_valid:
                self._update_report(module_name, 'Valid', details, api)
                return True
            else:
                self._update_report(module_name, 'INVALID', details, api)
                return False
        except Exception as e:
            self._update_report(module_name, 'ERROR', str(e), api)
            self.logger.error(f"An exception occurred during input validation for '{module_name}': {e}")
            return False

    def get_validation_report(self) -> pd.DataFrame:
        """Returns the final validation status DataFrame."""
        return self.report_df

    # --- Private INPUT Validation Methods ---

    # FIX: This method was missing from the previous version.
    def _validate_inputs_for_face_extraction(self, video_frames: List, dnn_net: Any, predictor: Any) -> (bool, str):
        if not isinstance(video_frames, list) or not video_frames:
            return False, "Input 'video_frames' must be a non-empty list."
        if dnn_net is None or predictor is None:
            return False, "Face detection models (dnn_net, predictor) are not loaded."
        return True, f"Ready to process {len(video_frames)} frames."

    def _validate_inputs_for_audio_features(self, audio_path: str, *models) -> (bool, str):
        if not os.path.exists(audio_path):
            return False, f"Audio file not found at: {audio_path}"
        return True, "Audio file and models are ready."

    def _validate_inputs_for_fer(self, faces: List, fps: int, model: Any) -> (bool, str):
        if not isinstance(faces, list): return False, "'faces' must be a list."
        if not isinstance(fps, int) or fps <= 0: return False, "'fps' must be a positive integer."
        if model is None: return False, "FER model is not loaded."
        if not faces: return True, "Input 'faces' list is empty. Skipping call."
        return True, f"Ready to process {len(faces)} faces."
    
    def _validate_inputs_for_valence_arousal(self, va_model: Any, feat_model: Any, faces: List, tensors: List) -> (bool, str):
        if va_model is None or feat_model is None: return False, "Valence/Arousal models are not loaded."
        if len(faces) != len(tensors): return False, f"Inconsistent inputs: {len(faces)} faces vs {len(tensors)} tensors."
        if not faces: return True, "Input 'faces' list is empty. Skipping call."
        return True, "Inputs are valid."

    def _validate_inputs_for_facs(self, model: Any, faces: List) -> (bool, str):
        if model is None: return False, "FACS model is not loaded."
        if not isinstance(faces, list): return False, "'faces' must be a list."
        if not faces: return True, "Input 'faces' list is empty. Skipping call."
        return True, f"Ready to process {len(faces)} faces."

    def _validate_inputs_for_behavioral_features(self, landmarks: List, sizes: List) -> (bool, str):
        if not isinstance(landmarks, list) or not isinstance(sizes, list):
            return False, "'landmarks' and 'sizes' must be lists."
        if len(landmarks) != len(sizes):
            return False, f"Inconsistent inputs: {len(landmarks)} landmarks vs {len(sizes)} sizes."
        if not landmarks: return True, "Input 'landmarks' is empty. Skipping call."
        return True, "Inputs are valid."

    def _validate_inputs_for_audio_analysis(self, audio_path: str) -> (bool, str):
        if not os.path.exists(audio_path):
            return False, f"Audio file for analysis not found at: {audio_path}"
        return True, "Audio file is ready."
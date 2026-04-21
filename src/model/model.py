from pathlib import Path
from typing import Optional, Tuple

import torch

from src.constants import MODEL_PATH
from src.model.architecture import StonesCNN
from src.services.audio_service import AudioService
from src.utils.logger import logger


class StonesModel:
    """Hot word detection model for inference."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.device = 'cpu'
        self.audio_service = AudioService()
        self.model = self._load_model(model_path or MODEL_PATH)
        logger.info("Model loaded")
    
    def _load_model(self, model_path: Path) -> StonesCNN:
        model = StonesCNN()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, audio_input) -> Tuple[Optional[int], Optional[float]]:
        """
        Predict hot word presence in audio.
        
        Args:
            audio_input: File path or numpy array
        
        Returns:
            Tuple of (prediction, confidence) or (None, None) on error
        """
        try:
            features = self.audio_service.extract_features(audio_input)
            if features is None:
                return None, None
            
            input_tensor = torch.FloatTensor(features).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][1].item()
            
            return prediction, confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, None
from pathlib import Path
from typing import Generator, Optional

import librosa
import numpy as np
import soundfile as sf

from src.constants import EXTRACTED_PATH, HOP_DURATION, SAMPLE_RATE, SAVE_OFFSET, WINDOW_DURATION
from src.utils.logger import logger


class AudioService:
    """Service for audio file processing and feature extraction."""
    
    def __init__(self, sr: int = SAMPLE_RATE, duration: float = WINDOW_DURATION,
                 n_mfcc: int = 40, n_fft: int = 400, hop_length: int = 160, n_mels: int = 40):
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def load_audio(self, audio_input, target_duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Load audio from file path or numpy array.
        
        Args:
            audio_input: File path or numpy array
            target_duration: Target duration in seconds (default: self.duration)
        
        Returns:
            Audio array or None on error
        """
        try:
            if isinstance(audio_input, (str, Path)):
                if not Path(audio_input).exists():
                    logger.error(f"File not found: {audio_input}")
                    return None
                y, _ = librosa.load(audio_input, sr=self.sr, duration=target_duration or self.duration)
            else:
                y = audio_input
            
            target_length = int(self.sr * (target_duration or self.duration))
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
            
            return y
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None
    
    def extract_features(self, audio_input) -> Optional[np.ndarray]:
        """
        Extract MFCC features with delta and delta-delta.
        
        Args:
            audio_input: File path or numpy array
        
        Returns:
            Features array with shape (1, time_frames, n_features) or None
        """
        y = self.load_audio(audio_input)
        if y is None:
            return None
        
        try:
            mfcc = librosa.feature.mfcc(
                y=y, sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
            features = features.astype(np.float32)
            return np.expand_dims(features, axis=0)
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def extract_windows(self, audio_path: Path, window_duration: float = WINDOW_DURATION,
                        hop_duration: float = HOP_DURATION,
                        save_offset: float = SAVE_OFFSET) -> Generator[tuple, None, None]:
        """
        Extract sliding windows from audio file.
        
        Args:
            audio_path: Path to audio file
            window_duration: Window duration in seconds
            hop_duration: Hop duration in seconds
            save_offset: Offset after detection window in seconds
        
        Yields:
            Tuple of (window_audio, start_time, end_time, save_fragment)
        """
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        window_samples = int(SAMPLE_RATE * window_duration)
        hop_samples = int(SAMPLE_RATE * hop_duration)
        offset_samples = int(SAMPLE_RATE * save_offset)
        total_needed = window_samples + offset_samples + window_samples
        
        for start in range(0, len(y) - total_needed + 1, hop_samples):
            end = start + window_samples
            window = y[start:end]
            save_start = end + offset_samples
            save_fragment = y[save_start:save_start + window_samples]
            start_time = start / SAMPLE_RATE
            end_time = end / SAMPLE_RATE
            yield window, start_time, end_time, save_fragment
    
    def save_audio(self, audio: np.ndarray, output_path: Path, sr: int = SAMPLE_RATE) -> bool:
        """
        Save audio array to file.
        
        Args:
            audio: Audio array
            output_path: Output file path
            sr: Sample rate
        
        Returns:
            True on success, False on error
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, sr)
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False

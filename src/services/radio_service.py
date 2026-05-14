import signal
import time
from pathlib import Path
from typing import Optional

import miniaudio
import numpy as np
import soundfile as sf

from src.constants import EXTRACTED_PATH, HOP_DURATION, RADIO_URL, SAMPLE_RATE, SAVE_OFFSET, THRESHOLD, WINDOW_DURATION
from src.utils.logger import logger


class RadioService:
    """Service for radio stream processing using miniaudio."""
    
    def __init__(self, model, sample_rate: int = SAMPLE_RATE):
        self.model = model
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * WINDOW_DURATION)
        self.hop_samples = int(sample_rate * HOP_DURATION)
    
    def _process_buffer(self, buffer: np.ndarray, stream_position: float,
                        threshold: float, output_dir: Path, found_count: int,
                        last_predict_log: float) -> tuple:
        """Process audio buffer with sliding window."""
        offset_samples = int(self.sample_rate * SAVE_OFFSET)
        while len(buffer) >= self.window_samples + offset_samples + self.window_samples:
            window = buffer[:self.window_samples]
            prediction, confidence = self.model.predict(window)
            logger.debug(f"Predict: pos={stream_position:.2f}s, class={prediction}, conf={confidence:.3f}")
            
            now = time.time()
            if now - last_predict_log >= 10.0:
                logger.info(f"Last predict: pos={stream_position:.2f}s, class={prediction}, conf={confidence:.3f}")
                last_predict_log = now
            
            if prediction == 1 and confidence >= threshold:
                found_count += 1
                save_start = self.window_samples + offset_samples
                save_fragment = buffer[save_start:save_start + self.window_samples]
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"stones_{timestamp}_{stream_position:.2f}s_conf{confidence:.2f}.wav"
                sf.write(output_dir / output_name, save_fragment, self.sample_rate)
                logger.info(f"DETECTED at {stream_position:.2f}s: confidence={confidence:.2f}, saved={output_name}")
            
            buffer = buffer[self.hop_samples:]
            stream_position += HOP_DURATION
        
        return buffer, stream_position, found_count, last_predict_log
    
    def listen(self, output_dir: Optional[Path] = None, threshold: float = THRESHOLD,
               url: str = RADIO_URL) -> int:
        """
        Listen to radio stream and detect hot words.
        
        Args:
            output_dir: Directory for saving detected fragments
            threshold: Confidence threshold for detection
            url: Radio stream URL
        
        Returns:
            Number of detected fragments
        """
        output_dir = Path(output_dir or EXTRACTED_PATH / "radio")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting radio stream: {url}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Threshold: {threshold}")
        
        buffer = np.array([], dtype=np.float32)
        found_count = 0
        stream_position = 0.0
        total_bytes = 0
        last_data_log = 0.0
        last_predict_log = 0.0
        
        logger.info("Warming up model...")
        self.model.predict(np.zeros(self.window_samples, dtype=np.float32))
        logger.info("Model ready. Listening...")
        
        client = None
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        while True:
            try:
                logger.info(f"Connecting to radio stream: {url}")
                client = miniaudio.IceCastClient(url)
                logger.info("Connected. Decoding stream...")
                
                stream = miniaudio.stream_any(
                    client,
                    source_format=miniaudio.FileFormat.MP3,
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=1,
                    sample_rate=self.sample_rate,
                    frames_to_read=self.hop_samples
                )
                
                for frames in stream:
                    samples = np.array(frames, dtype=np.float32) / 32768.0
                    buffer = np.concatenate([buffer, samples])
                    
                    total_bytes += len(frames) * 2
                    now = time.time()
                    if now - last_data_log >= 10.0:
                        logger.info(f"Receiving data: {total_bytes / 1024:.1f} KB total, buffer={len(buffer)} samples, position={stream_position:.2f}s")
                        last_data_log = now
                    
                    buffer, stream_position, found_count, last_predict_log = self._process_buffer(
                        buffer, stream_position, threshold, output_dir, found_count, last_predict_log
                    )
                    
                    if len(buffer) > self.window_samples * 10:
                        buffer = buffer[-self.window_samples * 5:]
                
                client.close()
                client = None
                logger.info("Stream ended. Reconnecting in 10s...")
                time.sleep(10.0)
            except KeyboardInterrupt:
                if client:
                    client.close()
                logger.info("Stopped by user")
                return found_count
            except Exception:
                logger.info("Connection error. Reconnecting in 10s...")
                time.sleep(10.0)

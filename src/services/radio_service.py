import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.constants import EXTRACTED_PATH, HOP_DURATION, RADIO_DURATION, RADIO_URL, SAMPLE_RATE, THRESHOLD, WINDOW_DURATION
from src.utils.logger import logger


class RadioService:
    """Service for radio stream processing."""
    
    def __init__(self, model, sample_rate: int = SAMPLE_RATE):
        self.model = model
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * WINDOW_DURATION)
        self.hop_samples = int(sample_rate * HOP_DURATION)
    
    def _build_ffmpeg_cmd(self, url: str) -> list:
        return [
            'ffmpeg',
            '-i', url,
            '-loglevel', 'warning',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', '1',
            '-'
        ]
    
    def _process_buffer(self, buffer: np.ndarray, stream_position: float,
                        threshold: float, output_dir: Path, found_count: int) -> tuple:
        """Process audio buffer with sliding window."""
        while len(buffer) >= self.window_samples:
            window = buffer[:self.window_samples]
            prediction, confidence = self.model.predict(window)
            
            logger.info(f"Position: {stream_position:.2f}s, Prediction: {prediction}, Confidence: {confidence:.2f}")
            
            if prediction == 1 and confidence >= threshold:
                found_count += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"stones_{timestamp}_{stream_position:.2f}s_conf{confidence:.2f}.wav"
                sf.write(output_dir / output_name, window, self.sample_rate)
                logger.info(f"Found at {stream_position:.2f}s: {output_name}")
            
            buffer = buffer[self.hop_samples:]
            stream_position += HOP_DURATION
        
        return buffer, stream_position, found_count
    
    def listen(self, output_dir: Optional[Path] = None, threshold: float = THRESHOLD,
               duration: float = RADIO_DURATION, url: str = RADIO_URL) -> int:
        """
        Listen to radio stream and detect hot words.
        
        Args:
            output_dir: Directory for saving detected fragments
            threshold: Confidence threshold for detection
            duration: Listening duration in seconds
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
        start_time = time.time()
        stream_position = 0.0
        chunk_size = 8000
        
        process = None
        try:
            logger.info("Starting ffmpeg decoder...")
            process = subprocess.Popen(
                self._build_ffmpeg_cmd(url),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            logger.info("Warming up model...")
            self.model.predict(np.zeros(self.window_samples, dtype=np.float32))
            logger.info("Model ready. Listening...")
            
            while True:
                if (time.time() - start_time) >= duration:
                    logger.info(f"Duration limit reached: {duration}s")
                    break
                
                if process.poll() is not None:
                    logger.error("ffmpeg process ended unexpectedly")
                    break
                
                raw_data = process.stdout.read(chunk_size)
                if not raw_data:
                    logger.warning("No data received from stream")
                    break
                
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, samples])
                
                buffer, stream_position, found_count = self._process_buffer(
                    buffer, stream_position, threshold, output_dir, found_count
                )
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        except FileNotFoundError:
            logger.error("ffmpeg not found")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            if process:
                process.terminate()
                process.wait()
            logger.info(f"Total found: {found_count} fragments")
            logger.info(f"Stream duration: {stream_position:.2f}s")
        
        return found_count
    
    async def listen_async(self, output_dir: Optional[Path] = None, threshold: float = THRESHOLD,
                           duration: float = RADIO_DURATION, url: str = RADIO_URL) -> int:
        """
        Async version of radio stream listening.
        
        Args:
            output_dir: Directory for saving detected fragments
            threshold: Confidence threshold for detection
            duration: Listening duration in seconds
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
        start_time = time.time()
        stream_position = 0.0
        chunk_size = 16000
        
        process = None
        try:
            logger.info("Starting ffmpeg decoder...")
            process = await asyncio.create_subprocess_exec(
                *self._build_ffmpeg_cmd(url),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            logger.info("Warming up model...")
            self.model.predict(np.zeros(self.window_samples, dtype=np.float32))
            logger.info("Model ready. Listening...")
            
            while True:
                if (time.time() - start_time) >= duration:
                    logger.info(f"Duration limit reached: {duration}s")
                    break
                
                if process.returncode is not None:
                    logger.error("ffmpeg process ended unexpectedly")
                    break
                
                try:
                    raw_data = await asyncio.wait_for(process.stdout.read(chunk_size), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                if not raw_data:
                    continue
                
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, samples])
                
                buffer, stream_position, found_count = self._process_buffer(
                    buffer, stream_position, threshold, output_dir, found_count
                )
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        except FileNotFoundError:
            logger.error("ffmpeg not found")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            if process:
                process.terminate()
                await process.wait()
            logger.info(f"Total found: {found_count} fragments")
            logger.info(f"Stream duration: {stream_position:.2f}s")
        
        return found_count

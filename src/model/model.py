import time
import torch
import argparse
import librosa
import numpy as np
import soundfile as sf
import asyncio
import sys
import subprocess
from pathlib import Path

from src.constants import MODEL_PATH, EXTRACTED_PATH, SAMPLE_RATE, WINDOW_DURATION, HOP_DURATION, RADIO_URL, THRESHOLD
from src.utils.logger import logger
from src.model.architecture import StonesCNN

class StonesModel:
    """Класс для инференса PyTorch модели"""

    def __init__(self):
        self.device = 'cpu'

        # Параметры MFCC (из ноутбука CNN_detector.ipynb)
        self.sr = 16000
        self.duration = 1.0
        self.n_mfcc = 40
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 40

        # Загрузка модели
        self.model = self._load_model()

        logger.info(f"Model loaded")

    def _load_model(self):
        """Загрузка PyTorch модели из .pt файла"""

        # Создание модели
        model = StonesCNN()

        # Загрузка весов
        model.load_state_dict(torch.load(MODEL_PATH))

        model.to(self.device)
        model.eval()

        return model

    def extract_features(self, audio_input):
        """
        Извлечение MFCC признаков

        Args:
            audio_input: путь к файлу или numpy array с аудио

        Returns:
            numpy array: (1, time_frames, n_features) - формат input модели
        """
        try:
            if isinstance(audio_input, (str, Path)):
                y, sr = librosa.load(audio_input, sr=self.sr, duration=self.duration)
            else:
                y = audio_input
                sr = self.sr

            target_length = int(self.sr * self.duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            mfcc = librosa.feature.mfcc(
                y=y, sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )

            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
            features = features.astype(np.float32)
            features = np.expand_dims(features, axis=0)

            return features

        except Exception as e:
            logger.error(f"Error while extracting features: {e}")
            return None

    def predict(self, audio_input):
        """
        Предсказание для аудиофайла или аудио массива

        Args:
            audio_input: путь к файлу или numpy array с аудио

        Returns:
            int: 1 - stones, 0 - not_stones
            float: уверенность
        """
        try:
            if isinstance(audio_input, (str, Path)) and not Path(audio_input).exists():
                logger.error(f"File not found: {audio_input}")
                return None, None

            features = self.extract_features(audio_input)
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
            logger.error(f"Error while predict processing: {e}")
            return None, None

    def extract_windows(self, audio_path):
        """
        Нарезка аудио на окна с оверлапом

        Yields:
            tuple: (window_audio, start_time, end_time)
        """
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        window_samples = int(SAMPLE_RATE * WINDOW_DURATION)
        hop_samples = int(SAMPLE_RATE * HOP_DURATION)

        for start in range(0, len(y) - window_samples + 1, hop_samples):
            end = start + window_samples
            window = y[start:end]
            start_time = start / SAMPLE_RATE
            end_time = end / SAMPLE_RATE
            yield window, start_time, end_time

    def process_audio_file(self, audio_path, output_dir=None, threshold=THRESHOLD):
        """
        Обработка аудиофайла: детекция и сохранение найденных фрагментов

        Args:
            audio_path: путь к аудиофайлу
            output_dir: директория для сохранения (по умолчанию data/extracted)
            threshold: порог уверенности для сохранения

        Returns:
            int: количество найденных фрагментов
        """
        if output_dir is None:
            output_dir = EXTRACTED_PATH

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio_path)
        y_full, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        found_count = 0
        logger.info(f"Processing: {audio_path.name}")

        for window, start_time, end_time in self.extract_windows(audio_path):
            prediction, confidence = self.predict(window)

            if prediction == 1 and confidence >= threshold:
                found_count += 1
                output_name = f"{audio_path.stem}_{start_time:.2f}-{end_time:.2f}.wav"
                output_path = output_dir / output_name

                window_samples = int(SAMPLE_RATE * WINDOW_DURATION)
                start_sample = int(start_time * SAMPLE_RATE)
                end_sample = start_sample + window_samples
                sf.write(output_path, y_full[start_sample:end_sample], sr)

                logger.info(f"Found: {output_name} (confidence: {confidence:.2f})")

        logger.info(f"Total found: {found_count} fragments")
        return found_count

    def listen_radio(self, output_dir=None, threshold=THRESHOLD, duration=None):
        """
        Прослушивание радио-потока и детекция hot-word

        Args:
            output_dir: директория для сохранения найденных фрагментов
            threshold: порог уверенности для сохранения
            duration: длительность прослушивания в секундах (None = бесконечно)

        Returns:
            int: количество найденных фрагментов
        """
        if output_dir is None:
            output_dir = EXTRACTED_PATH / 'radio'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting radio stream: {RADIO_URL}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Threshold: {threshold}")

        # Параметры
        window_samples = int(SAMPLE_RATE * WINDOW_DURATION)
        hop_samples = int(SAMPLE_RATE * HOP_DURATION)
        chunk_size = 8000  # 0.5 секунды аудио

        # Буфер
        buffer = np.array([], dtype=np.float32)
        found_count = 0
        start_time = time.time()
        stream_position = 0.0

        # ffmpeg команда для декодирования MP3 потока в PCM
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', RADIO_URL,
            '-loglevel', 'warning',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',
            '-'
        ]

        process = None

        try:
            logger.info("Starting ffmpeg decoder...")
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            # Прогрев модели
            logger.info("Warming up model...")
            self.predict(np.zeros(window_samples, dtype=np.float32))
            logger.info("Model ready. Listening...")

            while True:
                # Проверка длительности
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Duration limit reached: {duration}s")
                    break

                # Проверка что процесс жив
                if process.poll() is not None:
                    logger.error("ffmpeg process ended unexpectedly")
                    break

                # Читаем PCM данные
                raw_data = process.stdout.read(chunk_size)
                if not raw_data:
                    logger.warning("No data received from stream")
                    break

                # Конвертируем в numpy
                samples = np.frombuffer(raw_data, dtype=np.int16)
                samples = samples.astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, samples])

                # Обработка буфера скользящим окном
                while len(buffer) >= window_samples:
                    window = buffer[:window_samples]
                    prediction, confidence = self.predict(window)

                    logger.info(f"Position: {stream_position:.2f}s, Prediction: {prediction}, Confidence: {confidence:.2f}")

                    if prediction == 1 and confidence >= threshold:
                        found_count += 1
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_name = f"stones_{timestamp}_{stream_position:.2f}s_conf{confidence:.2f}.wav"
                        output_path = output_dir / output_name

                        sf.write(output_path, window, SAMPLE_RATE)
                        logger.info(f"Found at {stream_position:.2f}s: {output_name}")

                    buffer = buffer[hop_samples:]
                    stream_position += HOP_DURATION

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

    async def listen_radio_async(self, duration: float = None, threshold: float = None, output_dir: Path = None):
        """
        Асинхронная версия для радио-потока
        
        Args:
            duration: Длительность прослушивания в секундах (None = бесконечно)
            threshold: Порог уверенности для детекции
            output_dir: Директория для сохранения найденных фрагментов
        """
        threshold = threshold or THRESHOLD
        output_dir = output_dir or EXTRACTED_PATH / "radio"
        output_dir.mkdir(parents=True, exist_ok=True)

        window_samples = int(SAMPLE_RATE * WINDOW_DURATION)
        hop_samples = int(SAMPLE_RATE * HOP_DURATION)
        chunk_size = 16000  # 1 секунда аудио

        logger.info(f"Starting radio stream: {RADIO_URL}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Threshold: {threshold}")

        # ffmpeg команда для декодирования MP3 потока в PCM
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', RADIO_URL,
            '-loglevel', 'warning',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',
            '-'
        ]

        process = None
        buffer = np.array([], dtype=np.float32)
        found_count = 0
        start_time = time.time()
        stream_position = 0.0

        try:
            logger.info("Starting ffmpeg decoder...")
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Прогрев модели
            logger.info("Warming up model...")
            self.predict(np.zeros(window_samples, dtype=np.float32))
            logger.info("Model ready. Listening...")

            while True:
                # Проверка длительности
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Duration limit reached: {duration}s")
                    break

                # Проверка что процесс жив
                if process.returncode is not None:
                    logger.error("ffmpeg process ended unexpectedly")
                    break

                # Читаем данные с таймаутом
                try:
                    raw_data = await asyncio.wait_for(
                        process.stdout.read(chunk_size),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if not raw_data:
                    continue

                # Конвертируем в numpy
                samples = np.frombuffer(raw_data, dtype=np.int16)
                samples = samples.astype(np.float32) / 32768.0
                buffer = np.concatenate([buffer, samples])

                # Обработка буфера скользящим окном
                while len(buffer) >= window_samples:
                    window = buffer[:window_samples]
                    prediction, confidence = self.predict(window)

                    logger.info(f"Position: {stream_position:.2f}s, Prediction: {prediction}, Confidence: {confidence:.2f}")

                    if prediction == 1 and confidence >= threshold:
                        found_count += 1
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_name = f"stones_{timestamp}_{stream_position:.2f}s_conf{confidence:.2f}.wav"
                        output_path = output_dir / output_name

                        sf.write(output_path, window, SAMPLE_RATE)
                        logger.info(f"Found at {stream_position:.2f}s: {output_name}")

                    buffer = buffer[hop_samples:]
                    stream_position += HOP_DURATION

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

if __name__ == '__main__':
    try:
        # Init argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['extract', 'radio'])
        parser.add_argument('--file', required=False)

        args = parser.parse_args()

        # Set default paths
        if args.command == 'extract' and args.file is None:
            args.file = 'data/thanos_message.wav'

    except Exception as e:
        logger.error(f"An error occurred while parsing command: {e}")
        raise

    model = StonesModel()
    if args.command == 'extract':
        if args.file:
            model.process_audio_file(args.file)
        else:
            logger.error("No file specified for extract command")
    elif args.command == 'radio':
        # Use async version for better performance
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.run(model.listen_radio_async())
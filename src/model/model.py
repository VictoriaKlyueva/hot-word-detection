import torch
import argparse
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

from src.constants import MODEL_PATH, EXTRACTED_PATH, SAMPLE_RATE, WINDOW_DURATION, HOP_DURATION
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
                confidence = probabilities[0][prediction].item()

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

    def process_audio_file(self, audio_path, output_dir=None, threshold=0.98):
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
        pass
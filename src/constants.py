from pathlib import Path


MODEL_PATH = Path('models/best_model.pth')
EXTRACTED_PATH = Path('data/extracted')
SAMPLE_RATE = 16000
WINDOW_DURATION = 1.0
HOP_DURATION = 0.5
SAVE_OFFSET = 0.18
RADIO_URL = 'https://radio.maslovka-home.ru/thanosshows'
THRESHOLD = 0.88
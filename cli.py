#!.venv/bin/python

import argparse
import sys
from pathlib import Path

from src.constants import EXTRACTED_PATH, THRESHOLD, RADIO_URL
from src.model.model import StonesModel
from src.services.audio_service import AudioService
from src.services.radio_service import RadioService
from src.utils.logger import logger


def cmd_radio(args):
    """Listen to radio stream and detect hot words."""
    model = StonesModel()
    radio_service = RadioService(model)
    
    return radio_service.listen(
        output_dir=Path(args.output) if args.output else None,
        threshold=args.threshold,
        url=args.url
    )


def cmd_predict(args):
    """Detect hot words in audio file and save fragments."""
    model = StonesModel()
    audio_service = AudioService()
    
    audio_path = Path(args.file)
    output_dir = Path(args.output) if args.output else EXTRACTED_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {audio_path.name}")
    
    found_count = 0
    for window, start_time, end_time in audio_service.extract_windows(audio_path):
        prediction, confidence = model.predict(window)
        
        if prediction == 1 and confidence >= args.threshold:
            found_count += 1
            output_name = f"{audio_path.stem}_{start_time:.2f}-{end_time:.2f}.wav"
            audio_service.save_audio(window, output_dir / output_name)
            logger.info(f"Found: {output_name} (confidence: {confidence:.2f})")
    
    if found_count == 0:
        logger.info("No hot words found")
    else:
        logger.info(f"Total found: {found_count} fragments")
    
    return found_count


def main():
    parser = argparse.ArgumentParser(
        description="Hot word detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Radio command
    radio_parser = subparsers.add_parser("radio", help="Listen to radio stream and extract hot words from it")
    radio_parser.add_argument("--url", "-u", default=RADIO_URL, help="Radio stream URL")
    radio_parser.add_argument("--output", "-o", help="Output directory")
    radio_parser.add_argument("--threshold", "-t", type=float, default=THRESHOLD, help="Confidence threshold")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Detect hot words in audio file")
    predict_parser.add_argument("--file", "-f", default="data/thanos_message.wav", help="Path to audio file")
    predict_parser.add_argument("--output", "-o", help="Output directory")
    predict_parser.add_argument("--threshold", "-t", type=float, default=THRESHOLD,
                                help=f"Confidence threshold, default: {THRESHOLD}")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "radio": cmd_radio,
        "predict": cmd_predict
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

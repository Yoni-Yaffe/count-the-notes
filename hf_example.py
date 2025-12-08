"""
Example script demonstrating how to use CountEM models from Hugging Face Hub.

This script shows how to:
1. Load a pre-trained CountEM model from Hugging Face Hub
2. Load an audio file
3. Transcribe the audio to MIDI

Requirements:
    pip install huggingface_hub soundfile

Usage:
    python hf_example.py path/to/audio.flac output.mid
"""

import sys
import argparse
from pathlib import Path
from onsets_and_frames.hf_model import CountEMModel
import soundfile as sf
from onsets_and_frames.constants import SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to MIDI using CountEM models from Hugging Face Hub"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to input audio file (must be 16kHz)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save output MIDI file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="a different value",
        choices=["Yoni-Yaffe/countem-musicnet", "Yoni-Yaffe/countem-synth"],
        help="Model to use for transcription (default: countem-musicnet)",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Threshold for onset detection (default: 0.5)",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.5,
        help="Threshold for frame detection (default: 0.5)",
    )

    args = parser.parse_args()

    # Validate input file exists
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("CountEM - Automatic Music Transcription")
    print(f"{'=' * 60}\n")

    # Step 1: Load the model from Hugging Face Hub
    print(f"Loading model: {args.model}")
    print("(This may take a moment on first run as the model downloads)")
    model = CountEMModel.from_pretrained(args.model)
    print("✓ Model loaded successfully\n")

    # Step 2: Load audio file
    print(f"Loading audio: {audio_path}")
    try:
        audio, sr = sf.read(str(audio_path), dtype="float32")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)

    # Check sample rate
    if sr != SAMPLE_RATE:
        print(
            f"\nWarning: Audio sample rate is {sr}Hz, but model expects {SAMPLE_RATE}Hz"
        )
        print(f"Please resample your audio to {SAMPLE_RATE}Hz for best results.")
        print("You can use librosa or ffmpeg to resample:")
        print(f"  ffmpeg -i {audio_path} -ar {SAMPLE_RATE} output_16k.flac")
        sys.exit(1)

    print(f"✓ Audio loaded: {len(audio) / sr:.2f} seconds, {sr}Hz\n")

    # Step 3: Transcribe to MIDI
    print("Transcribing to MIDI...")
    print(f"  Onset threshold: {args.onset_threshold}")
    print(f"  Frame threshold: {args.frame_threshold}")

    try:
        model.transcribe_to_midi(
            audio,
            args.output_path,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
        )
        print("\n✓ Transcription complete!")
        print(f"  MIDI saved to: {args.output_path}")
    except Exception as e:
        print(f"\nError during transcription: {e}")
        sys.exit(1)

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()

"""
Hugging Face Hub-compatible wrapper for CountEM music transcription models.
"""
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import torch
import soundfile as sf
from huggingface_hub import PyTorchModelHubMixin

from onsets_and_frames.transcriber import OnsetsAndFrames
from onsets_and_frames.mel import MelSpectrogram
from onsets_and_frames.midi_utils import frames2midi
from onsets_and_frames.constants import (
    N_MELS,
    MIN_MIDI,
    MAX_MIDI,
    HOP_LENGTH,
    SAMPLE_RATE,
    WINDOW_LENGTH,
    MEL_FMIN,
    MEL_FMAX,
)


class CountEMModel(
    OnsetsAndFrames,
    PyTorchModelHubMixin,
    # Optional metadata that gets pushed to model card
    library_name="countem",
    tags=["audio", "music-transcription", "automatic-music-transcription", "midi"],
    license="cc-by-4.0",
    repo_url="https://github.com/Yoni-Yaffe/count-the-notes",
    paper_url="https://arxiv.org/abs/2511.14250",
):
    """
    Hugging Face Hub-compatible wrapper for CountEM automatic music transcription models.

    This model performs automatic music transcription (AMT) from audio to MIDI.
    It uses the Onsets & Frames architecture trained with the CountEM framework,
    which enables training with weak, unordered note count histograms.

    Example usage:
        ```python
        from onsets_and_frames.hf_model import CountEMModel
        import soundfile as sf

        # Load model from Hub
        model = CountEMModel.from_pretrained("Yoni-Yaffe/countem-musicnet")

        # Load audio (must be 16kHz)
        audio, sr = sf.read("audio.flac")
        assert sr == 16000, "Audio must be 16kHz"

        # Transcribe to MIDI
        model.transcribe_to_midi(audio, "output.mid")
        ```

    Args:
        model_complexity: Complexity multiplier for the model (default: 64)
        onset_complexity: Complexity multiplier for onset stack (default: 1.5)
        n_instruments: Number of instruments to transcribe (default: 1)
    """

    def __init__(
        self,
        model_complexity: int = 64,
        onset_complexity: float = 1.5,
        n_instruments: int = 1,
        **kwargs
    ):
        # Initialize the base OnsetsAndFrames model
        n_keys = MAX_MIDI - MIN_MIDI + 1
        OnsetsAndFrames.__init__(
            self,
            input_features=N_MELS,
            output_features=n_keys,
            model_complexity=model_complexity,
            onset_complexity=onset_complexity,
            n_instruments=n_instruments,
        )

        # Store config for HF Hub
        self.config = {
            "model_complexity": model_complexity,
            "onset_complexity": onset_complexity,
            "n_instruments": n_instruments,
            "n_mels": N_MELS,
            "n_keys": n_keys,
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
        }

        # Add mel spectrogram as a submodule for proper device management
        # This ensures the mel transform moves with the model when calling .to(device)
        self.melspectrogram = MelSpectrogram(
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            filter_length=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        )

    def forward(self, audio: Union[np.ndarray, torch.Tensor]):
        """
        Forward pass that accepts raw audio waveforms.

        Unlike the parent OnsetsAndFrames which expects mel spectrograms,
        this forward method accepts raw audio and converts it internally.

        Args:
            audio: Raw audio waveform, shape (batch, n_samples) or (n_samples,)
                   Should be normalized to [-1, 1] or will be normalized automatically

        Returns:
            Tuple of (onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred)
        """
        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure audio is in range [-1, 1]
        if audio.dtype == torch.int16:
            audio = audio.float() / 32768.0
        elif audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = next(self.parameters()).device
        audio = audio.to(device)

        # Remove last sample (consistent with training code)
        # This is done in dataset.py, transcriber.py, and inference.py
        audio = audio[:, :-1]

        mel = self.melspectrogram(audio)

        # Transpose to (batch, time, features) format expected by parent model
        mel = mel.transpose(-1, -2)

        return super().forward(mel)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transcribe audio to note predictions.

        Automatically handles long audio by splitting into segments (max 5 minutes each)
        to avoid memory issues.

        Args:
            audio: Audio waveform, shape (n_samples,), normalized to [-1, 1]
            onset_threshold: Threshold for onset detection (default: 0.5)
            frame_threshold: Threshold for frame detection (default: 0.5)

        Returns:
            Tuple of (onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred)
            All are numpy arrays of shape (n_frames, 88) except velocity which may vary
        """
        self.eval()

        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure audio is 1D (convert stereo to mono if needed)
        if audio.dim() > 1:
            # If stereo or multi-channel, take mean across channels
            audio = audio.mean(dim=-1 if audio.shape[-1] <=2 else 0)

        # Normalize audio
        if audio.dtype == torch.int16:
            audio = audio.float() / 32768.0
        elif audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        device = next(self.parameters()).device
        audio = audio.to(device)

        # Handle long audio by segmenting
        MAX_TIME = 5 * 60 * SAMPLE_RATE  # 5 minutes
        audio_len = len(audio)

        if audio_len > MAX_TIME:
            # Split into segments
            n_segments = int(np.ceil(audio_len / MAX_TIME))
            seg_len = MAX_TIME

            onset_preds = []
            offset_preds = []
            activation_preds = []
            frame_preds = []
            velocity_preds = []

            for i_s in range(n_segments):
                start = i_s * seg_len
                end = min((i_s + 1) * seg_len, audio_len)
                segment = audio[start:end]

                # Forward pass on segment
                onset_seg, offset_seg, activation_seg, frame_seg, velocity_seg = self(segment)

                onset_preds.append(onset_seg)
                offset_preds.append(offset_seg)
                activation_preds.append(activation_seg)
                frame_preds.append(frame_seg)
                velocity_preds.append(velocity_seg)

            # Concatenate along time dimension (dim=1)
            onset_pred = torch.cat(onset_preds, dim=1)
            offset_pred = torch.cat(offset_preds, dim=1)
            activation_pred = torch.cat(activation_preds, dim=1)
            frame_pred = torch.cat(frame_preds, dim=1)
            velocity_pred = torch.cat(velocity_preds, dim=1)
        else:
            # Short audio, process directly
            onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred = self(audio)

        # Convert to numpy and remove batch dimension
        onset_pred = onset_pred.squeeze(0).cpu().numpy()
        offset_pred = offset_pred.squeeze(0).cpu().numpy()
        activation_pred = activation_pred.squeeze(0).cpu().numpy()
        frame_pred = frame_pred.squeeze(0).cpu().numpy()
        velocity_pred = velocity_pred.squeeze(0).cpu().numpy()

        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def transcribe_to_midi(
        self,
        audio: Union[np.ndarray, torch.Tensor, str, Path],
        output_path: Union[str, Path],
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.5,
    ) -> None:
        """
        Transcribe audio to MIDI file.

        Args:
            audio: Audio waveform, numpy array, torch tensor, or path to audio file
            output_path: Path to save MIDI file
            onset_threshold: Threshold for onset detection (default: 0.5)
            frame_threshold: Threshold for frame detection (default: 0.5)
        """
        # Load audio from file if path is provided
        if isinstance(audio, (str, Path)):
            audio, sr = sf.read(audio, dtype="float32")
            if sr != SAMPLE_RATE:
                raise ValueError(
                    f"Audio must be {SAMPLE_RATE}Hz, got {sr}Hz. "
                    f"Please resample to {SAMPLE_RATE}Hz first."
                )

        # Get predictions
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self.transcribe(
            audio, onset_threshold, frame_threshold
        )

        # Default instrument mapping (piano)
        inst_mapping = {0: 0}  # instrument 0 -> MIDI program 0 (Acoustic Grand Piano)

        # Convert predictions to MIDI
        frames2midi(
            str(output_path),
            onset_pred,
            frame_pred,
            velocity_pred,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            scaling=HOP_LENGTH / SAMPLE_RATE,
            inst_mapping=inst_mapping,
        )

    def to_legacy(self) -> OnsetsAndFrames:
        """
        Convert this HuggingFace-compatible model to a legacy OnsetsAndFrames instance.

        This is useful for:
        - Fine-tuning models downloaded from HuggingFace Hub using existing training code
        - Using HF models with existing inference scripts that expect OnsetsAndFrames

        The legacy model will use the global melspectrogram from mel.py instead of
        the instance-specific one in this model.

        Returns:
            OnsetsAndFrames instance with copied weights
        """
        # Create legacy model with same architecture
        legacy_model = OnsetsAndFrames(
            input_features=self.config['n_mels'],
            output_features=self.config['n_keys'],
            model_complexity=self.config['model_complexity'],
            onset_complexity=self.config['onset_complexity'],
            n_instruments=self.config['n_instruments']
        )

        # Get the state dict and filter out melspectrogram keys
        state_dict = self.state_dict()
        legacy_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('melspectrogram.')}

        # Copy state dict (only model weights, not mel spectrogram)
        # The legacy model will use the global melspectrogram
        legacy_model.load_state_dict(legacy_state_dict)

        return legacy_model

    @classmethod
    def from_legacy_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        **kwargs
    ) -> "CountEMModel":
        """
        Load a model from a legacy checkpoint (saved with torch.save(model)).

        This is useful for converting old checkpoints to the new HF-compatible format.

        Args:
            checkpoint_path: Path to the legacy .pt checkpoint file
            **kwargs: Additional arguments for model initialization

        Returns:
            CountEMModel instance with loaded weights
        """
        # Load the legacy checkpoint
        legacy_model = torch.load(checkpoint_path, map_location="cpu")

        # Extract configuration from the loaded model
        # Infer model_complexity from the model structure
        # ConvStack.cnn[0] is the first Conv2d layer with out_channels = model_size // 16
        first_conv_channels = legacy_model.offset_stack[0].cnn[0].out_channels
        model_size = first_conv_channels * 16
        model_complexity = model_size // 16

        # Infer onset_complexity
        onset_first_conv_channels = legacy_model.onset_stack[0].cnn[0].out_channels
        onset_model_size = onset_first_conv_channels * 16
        onset_complexity = onset_model_size / model_size

        # Infer n_instruments from output layer
        # onset_stack[2] is the Linear layer
        onset_out_features = legacy_model.onset_stack[2].out_features
        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_instruments = onset_out_features // n_keys

        # Create new model with the same configuration
        model = cls(
            model_complexity=model_complexity,
            onset_complexity=onset_complexity,
            n_instruments=n_instruments,
            **kwargs
        )

        # Copy the state dict (strict=False because new model has melspectrogram submodule)
        model.load_state_dict(legacy_model.state_dict(), strict=False)

        return model

import torch
import torch.nn.functional as F
from torch import nn
from .mel import melspectrogram
from .lstm import BiLSTM
from onsets_and_frames.constants import MAX_MIDI, MIN_MIDI, N_KEYS

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, positive_weight=2., inv_positive_weight=2.,
                     with_onset_mask=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        if 'velocity' in batch:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        onset_mask = 1. * onset_label
        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if with_onset_mask:
            if 'onset_mask' in batch:
                onset_mask = onset_mask * batch['onset_mask']
        # if 'onset_mask' in batch:
        #     onset_mask += batch['onset_mask']


        offset_mask = 1. * offset_label
        offset_positive_weight = 2.
        offset_mask *= (offset_positive_weight - 1)
        offset_mask += 1.

        frame_mask = 1. * frame_label
        frame_positive_weight = 2.
        frame_mask *= (frame_positive_weight - 1)
        frame_mask += 1.

        for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


#   same implementation as OnsetsAndFrames, but with only onset stack
class OnsetsNoFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        return onset_pred

    def run_on_batch(self, batch, parallel_model=None, positive_weight=2., inv_positive_weight=2., with_onset_mask=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        if not parallel_model:
            onset_pred = self(mel)
        else:
            onset_pred = parallel_model(mel)


        predictions = {
            'onset': onset_pred,
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
        }

        onset_mask = 1. * onset_label
        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if with_onset_mask:
            if 'onset_mask' in batch:
                onset_mask = onset_mask * batch['onset_mask']

        losses['loss/onset'] = (onset_mask * losses['loss/onset']).mean()

        return predictions, losses


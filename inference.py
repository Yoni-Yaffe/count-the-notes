
from onsets_and_frames import *
import soundfile
from torch.nn import DataParallel

from onsets_and_frames.midi_utils import frames2midi
import sys
import yaml
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and convert audio to MIDI")

    parser.add_argument('--ckpt', type=str, required=True, help='Path to transcriber checkpoint')
    parser.add_argument('--audio-files-dir', type=str, required=True, help='Directory containing FLAC files for inference')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory for saving MIDI files')
    
    parser.add_argument('--save-onsets-and-frames', action='store_true', help='Save raw onset/frame predictions as .npy files')

    return parser.parse_args()

def set_diff(model, diff=True):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = diff


def load_audio(flac):
    audio, sr = soundfile.read(flac, dtype='int16')
    if len(audio.shape) == 2:
        audio = audio.astype(float).mean(axis=1)
    else:
        audio = audio.astype(float)
    audio = audio.astype(np.int16)
    print('audio len', len(audio))
    assert sr == SAMPLE_RATE
    audio = torch.ShortTensor(audio)
    return audio


def inference_single_flac(transcriber, flac_path, inst_mapping, out_dir, save_onsets_and_frames=False):
    audio = load_audio(flac_path)
    audio_inp = audio.float() / 32768.
    MAX_TIME = 5 * 60 * SAMPLE_RATE  # 5 minutes
    audio_inp_len = len(audio_inp)
    print("audio inp len", audio_inp_len)
    if audio_inp_len > MAX_TIME:
        n_segments = int(np.ceil(audio_inp_len / MAX_TIME)) 
        print('long audio, splitting to {} segments'.format(n_segments))
        seg_len = MAX_TIME
        onsets_preds = []
        offset_preds = []
        frame_preds = []
        
        for i_s in range(n_segments):
            curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
            curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
            curr_onset_pred, curr_offset_pred, _, curr_frame_pred, *_ = transcriber(curr_mel)
            onsets_preds.append(curr_onset_pred)
            offset_preds.append(curr_offset_pred)
            frame_preds.append(curr_frame_pred)
            
        onset_pred = torch.cat(onsets_preds, dim=1)
        offset_pred = torch.cat(offset_preds, dim=1)
        frame_pred = torch.cat(frame_preds, dim=1)
        
            
    else:
        print("didn't have to split")
        audio_inp = audio_inp.unsqueeze(0).cuda()
        mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, *_ = transcriber(mel)
                
    onset_pred = onset_pred.detach().squeeze().cpu()
    frame_pred = frame_pred.detach().squeeze().cpu()
    

    onset_pred_np = onset_pred.numpy()
    frame_pred_np = frame_pred.numpy()
        
    save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '.mid'))
    
    print("onset pred shape", onset_pred_np.shape)

    onset_threshold = 0.5
    scaling = constants.HOP_LENGTH / SAMPLE_RATE
    print("Hop size", constants.HOP_LENGTH)
    print("onset threshold", onset_threshold)
    print("scaling", scaling)
    sys.stdout.flush()
    frames2midi(save_path,
                onset_pred_np, frame_pred_np,
                64. * onset_pred_np,
                inst_mapping=inst_mapping, onset_threshold=onset_threshold, scaling=scaling)

    if save_onsets_and_frames:
        onset_save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '_onset_pred.npy'))
        frame_save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '_frame_pred.npy'))
        np.save(onset_save_path, onset_pred_np)
        np.save(frame_save_path, frame_pred_np)
    
    
    print(f"saved midi to {save_path}")
    return save_path

def inference_multiple_flacs(transcriber_ckpt, flac_dir, config, save_onsets_and_frames=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device {device}")
    
    transcriber = torch.load(transcriber_ckpt).to(device)
    set_diff(transcriber.frame_stack, False)
    set_diff(transcriber.offset_stack, False)
    set_diff(transcriber.combined_stack, False)
    if hasattr(transcriber, 'velocity_stack'):
        set_diff(transcriber.velocity_stack, False)
    print("transcriber: ", transcriber)

    parallel_transcriber = DataParallel(transcriber)

    transcriber.zero_grad()
    transcriber.eval()
    parallel_transcriber.eval()
    flac_path_list = [os.path.join(flac_dir, f) for f in os.listdir(flac_dir) if f.endswith('.flac')]
    results_dir = os.path.join(config['out_dir'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    with torch.no_grad():
        for flac_path in flac_path_list:
            print(f"processing {flac_path}")
            inference_single_flac(transcriber=parallel_transcriber,
                                  flac_path=flac_path,
                                  inst_mapping=[0], 
                                  out_dir=results_dir,
                                  save_onsets_and_frames=save_onsets_and_frames)


def generate_labels_wrapper(config_input):
    if isinstance(config_input, argparse.Namespace):
        config = vars(config_input)
    else:
        config = config_input

    config['out_dir'] = config.get('out_dir', './results')
    ckpt = config['ckpt']
    flac_dir = config['audio_files_dir']
    save_onsets_and_frames = config.get('save_onsets_and_frames', False)


    inference_multiple_flacs(
        transcriber_ckpt=ckpt,
        flac_dir=flac_dir,
        config=config,
        save_onsets_and_frames=save_onsets_and_frames
    )


if __name__ == '__main__':
    args = parse_args()
    generate_labels_wrapper(args)
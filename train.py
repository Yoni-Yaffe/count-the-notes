import os
from datetime import datetime
import numpy as np
from torch.nn.utils import clip_grad_norm_
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights, load_weights_pop, modulated_load_weights
import time
from conversion_maps import pop_conversion_map, classic_conversion_map, constant_conversion_map
import sys
import yaml
import random


def set_diff(model, diff=True):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = diff

def worker_init_fn(worker_id):
    # 1) grab the unique seed PyTorch assigned to this worker
    seed = torch.initial_seed() % (2**32)
    # 2) print it out for debugging
    print(f"[worker {worker_id} | PID {os.getpid()}] seed = {seed}")
    # 3) seed Python + NumPy
    random.seed(seed)
    np.random.seed(seed)
    


def train(logdir, device, iterations, checkpoint_interval, batch_size, sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt, config: dict):
    
    print(f"config -  {config}")
    print("Cuda is available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Start time: ", datetime.now())
    print("device ", device)
    print("device name", torch.cuda.get_device_name(device=device))
    # Place holders
    print("HOP LENGTH", HOP_LENGTH)
    print("SEQUENCE LENGTH", sequence_length)
    
    onset_precision = None
    onset_recall = None
    pitch_onset_precision = None
    pitch_onset_recall = None
    loss_list = []
    iter_list = []
    
    if config.get('seed', None) is not None:
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        print(f"seed is set to {seed}")

    total_run_1 = time.time()
    print(f"device {device}")
    os.makedirs(logdir, exist_ok=True)
    # n_weight = 1 if HOP_LENGTH == 512 else 2
    n_weight = config['n_weight']
    dataset_name = config['dataset_name']
    
    # train_data_path = f'/vol/scratch/jonathany/datasets/{dataset_name}/noteEM_audio'
    train_data_path = f'/home/dcor/jonathany/datasets/{dataset_name}/noteEM_audio'
    if 'train_data_path' in config:
        train_data_path = os.path.join(config['train_data_path'], dataset_name, 'noteEM_audio')
        


    if 'labels_dir_path' in config:
        labels_path = config['labels_dir_path']
    else:
        labels_path = os.path.join(logdir, 'NoteEm_labels')

    print("Lables path:", labels_path)


    os.makedirs(labels_path, exist_ok=True)
    score_log_path = os.path.join(logdir, "score_log.txt")
    with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
        fp.write(f"Parameters:\ndevice: {device}, iterations: {iterations}, checkpoint_interval: {checkpoint_interval},"
                 f" batch_size: {batch_size}, sequence_length: {sequence_length}, learning_rate: {learning_rate}, "
                 f"learning_rate_decay_steps: {learning_rate_decay_steps}, clip_gradient_norm: {clip_gradient_norm}, "
                 f"epochs: {epochs}, transcriber_ckpt: {transcriber_ckpt}, multi_ckpt: {multi_ckpt}, n_weight: {n_weight}\n")
    
    if 'groups' in config:
        train_groups = config['groups']
    else:
        train_groups = [dataset_name]
    
    conversion_map = None
    if 'use_pop_conversion_map' in config and config['use_pop_conversion_map']:
        conversion_map = pop_conversion_map.conversion_map
    elif 'use_classic_conversion_map' in config and config['use_classic_conversion_map']:
        conversion_map = classic_conversion_map.conversion_map
    elif 'use_constant_conversion_map' in config and config['use_constant_conversion_map']:
        conversion_map = constant_conversion_map.conversion_map
    
    instrument_map = None
        
    dataset = EMDATASET(audio_path=train_data_path,
                        tsv_path=config['tsv_dir'],
                           labels_path=labels_path,
                           groups=train_groups,
                            sequence_length=sequence_length,
                            seed=42,
                           device=DEFAULT_DEVICE,
                            instrument_map=instrument_map,
                            conversion_map=conversion_map,
                            pitch_shift=config['pitch_shift'],
                            pitch_shift_limit=config.get('pitch_shift_limit', 5),
                            keep_eval_files=config.get('make_evaluation', False),
                            evaluation_list=config.get('evaluation_list', None),
                            only_eval= (iterations == 0),
                            save_to_memory=config.get('save_to_memory', False),
                            smooth_labels=config.get('smooth_labels', False),
                            use_onset_mask=config.get('use_onset_mask', False),
                        )
    if iterations > 0:
        print('len dataset', len(dataset), len(dataset.data))

    #####
    if not multi_ckpt:
        model_complexity = 64
        onset_complexity = 1.5
        # We create a new transcriber with N_KEYS classes for each instrument:
        transcriber = OnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                    model_complexity,
                                onset_complexity=onset_complexity, n_instruments=1)
    else:
        transcriber = torch.load(transcriber_ckpt)

    print("HOP LENGTH", constants.HOP_LENGTH, HOP_LENGTH)    
    
    if hasattr(transcriber, 'onset_stack'):
        set_diff(transcriber.onset_stack, True)
    if hasattr(transcriber, 'frame_stack'):
        set_diff(transcriber.offset_stack, False)
    if hasattr(transcriber, 'combined_stack'):
        set_diff(transcriber.combined_stack, False)
    if hasattr(transcriber, 'velocity_stack'):
        set_diff(transcriber.velocity_stack, False)

    prev_transcriber = None
    if config.get('onset_no_frames_model', False):
        model_complexity = 64
        onset_complexity = 1.5
        transcriber2 = OnsetsNoFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                        model_complexity,
                                    onset_complexity=onset_complexity, n_instruments=1).to(device)
        transcriber2.onset_stack.load_state_dict(transcriber.onset_stack.state_dict())
        prev_transcriber =  transcriber
        transcriber = transcriber2
    print("transcriber", transcriber)
    parallel_transcriber = DataParallel(transcriber)
    parallel_transcriber = parallel_transcriber.to(device)
    optimizer = torch.optim.Adam(list(transcriber.parameters()), lr=learning_rate, weight_decay=0)
    transcriber.zero_grad()
    optimizer.zero_grad()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()

        POS = 1.1 # Pseudo-label positive threshold (value > 1 means no pseudo label).
        NEG = -0.1 # Pseudo-label negative threshold (value < 0 means no pseudo label).
        if config['psuedo_labels']:
            POS = 0.5
            NEG = 0.01
            
        counting_window = config['counting_window']
        counting_window_length = counting_window
        # counting_window_length = counting_window * SAMPLE_RATE // HOP_LENGTH
        transcriber.eval()
        to_save_dir  = os.path.join(logdir, 'alignments') if config.get('save_updated_labels_midis', False) else None
        with torch.no_grad():
            dataset.update_pts_counting(parallel_transcriber,
                counting_window_length, 
                POS=POS,
                NEG=NEG,
                FRAME_POS=0.5,
                to_save=to_save_dir,
                first=epoch == 1,
                update=config.get('update_labels', False),
                BEST_DIST=config.get('best_dist_update', False),
                peak_size=config.get('peak_size', 3),
                BEST_DIST_VEC=config.get('best_dist_vec', False), 
                counting_window_hop=config.get('counting_window_hop', 0), 
            )
        
        num_workers = config.get('num_workers', 0)
        print("num workers:", num_workers)
        if num_workers > 0:
            loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=config.get('num_workers', 0), pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
        else:
            loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
        
        total_loss = []
        curr_loss = []
        transcriber.train()
            
        onset_total_tp = 0.
        onset_total_pp = 0.
        onset_total_p = 0.

        torch.cuda.empty_cache()
        loader_cycle = cycle(loader)
        time_start = time.time()
        for iteration in tqdm(range(1, iterations + 1), desc=f"Train Loop Epoch {epoch}"):
            curr_loader = loader_cycle
            batch = next(curr_loader)
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad()
            transcription, transcription_losses = transcriber.run_on_batch(batch, parallel_model=parallel_transcriber, positive_weight=n_weight,
                                                                           inv_positive_weight=n_weight, with_onset_mask=config.get('with_onset_mask', False))
            
            onset_pred = transcription['onset'].detach() > 0.5
            onset_total_pp += onset_pred
            onset_tp = onset_pred * batch['onset'].detach()
            onset_total_tp += onset_tp
            onset_total_p += batch['onset'].detach()

            onset_recall = (onset_total_tp.sum() / onset_total_p.sum()).item()
            onset_precision = (onset_total_tp.sum() / onset_total_pp.sum()).item()
            pitch_onset_recall = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()).item()
            pitch_onset_precision = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_pp[..., -N_KEYS:].sum()).item()
            
            

            transcription_loss = transcription_losses['loss/onset']
            
            loss = transcription_loss
            curr_loss_value = loss.item()
            loss.backward()

            if clip_gradient_norm:
                clip_grad_norm_(transcriber.parameters(), clip_gradient_norm)

            optimizer.step()
            total_loss.append(curr_loss_value)
            curr_loss.append(curr_loss_value)
            # print(f"avg loss: {sum(total_loss) / len(total_loss):.5f} current loss: {total_loss[-1]:.5f} Onset Precision: {onset_precision:.3f} Onset Recall {onset_recall:.3f} "
            #       f"Pitch Onset Precision: {pitch_onset_precision:.3f} Pitch Onset Recall {pitch_onset_recall:.3f}")
            if epochs == 1 and iteration % 20000 == 1:
                transriber_path = os.path.join(logdir, 'transcriber_iteration_{}.pt'.format(iteration))
                if config.get('onset_no_frames_model', False):
                    prev_transcriber.onset_stack.load_state_dict(transcriber.onset_stack.state_dict())
                    torch.save(prev_transcriber, transriber_path)
                else:
                    torch.save(transcriber, transriber_path)
                torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
                torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(iteration)))
            
            if epochs == 1 and iteration % 1000 == 1:
                score_msg = f"iteration {iteration:06d} loss: {np.mean(curr_loss):.5f} Onset Precision:  {onset_precision:.3f} " \
                    f"Onset Recall {onset_recall:.3f} Pitch Onset Precision:  {pitch_onset_precision:.3f} " \
                    f"Pitch Onset Recall  {pitch_onset_recall:.3f}\n"
                print(score_msg)
                loss_list.append(np.mean(curr_loss))
                iter_list.append(iteration)
                curr_loss = []
                    
                onset_total_tp = 0.
                onset_total_pp = 0.
                onset_total_p = 0.
            elif epochs != 1 and iteration % 1000 == 1:
                loss_list.append(np.mean(curr_loss))
                iter_list.append(iteration + iterations * (epoch - 1))
                curr_loss = []
                score_msg = f"iteration {iteration:06d} loss: {np.mean(curr_loss):.5f} Onset Precision:  {onset_precision:.3f} " \
                    f"Onset Recall {onset_recall:.3f} Pitch Onset Precision:  {pitch_onset_precision:.3f} " \
                    f"Pitch Onset Recall  {pitch_onset_recall:.3f}\n"
                print(score_msg)
                
            if epochs == 1 and iteration % 2500 == 0:
                transcriber_path = os.path.join(logdir, 'transcriber_ckpt.pt'.format(iteration))
                if config.get('onset_no_frames_model', False):
                    prev_transcriber.onset_stack.load_state_dict(transcriber.onset_stack.state_dict())
                    torch.save(prev_transcriber, transcriber_path)
                else:
                    torch.save(transcriber, transcriber_path)



        time_end = time.time()
        score_msg = f"epoch {epoch:02d} loss: {sum(total_loss) / len(total_loss):.5f} Onset Precision:  {onset_precision:.3f} " \
                    f"Onset Recall {onset_recall:.3f} Pitch Onset Precision:  {pitch_onset_precision:.3f} " \
                    f"Pitch Onset Recall  {pitch_onset_recall:.3f} time label update: {time.strftime('%M:%S', time.gmtime(time_end - time_start))}\n"

        save_condition = epoch % checkpoint_interval == 1 or checkpoint_interval == 1
        if save_condition and epochs != 1:
            torch.save(transcriber, os.path.join(logdir, 'transcriber_{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(epoch)))
        print(score_msg)
        
    total_run_2 = time.time()
    print(f"Total Runtime: {time.strftime('%H:%M:%S', time.gmtime(total_run_2 - total_run_1))}\n")
    
    # keep last optimized state
    torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
    torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'))        
            
    
    
    
def train_wrapper(yaml_config: dict, logdir):
    config = yaml_config['train_params']
    transcriber_ckpt = config['transcriber_ckpt']
    multi_ckpt = config['multi_ckpt']
    checkpoint_interval = config['checkpoint_interval']
    batch_size = config['batch_size']
    iterations = config['iterations']
    learning_rate = config['learning_rate']
    learning_rate_decay_steps = config['learning_rate_decay_steps']
    clip_gradient_norm = config['clip_gradient_norm']
    epochs = config['epochs']
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sequence_length = constants.SEQ_LEN 


    train(logdir, device, iterations, checkpoint_interval, batch_size, sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt, config)
    
    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_path = 'config.yaml'
    else:
        logdir = sys.argv[1]
        yaml_path = os.path.join(logdir, 'run_config.yaml')

    with open(yaml_path, 'r') as fp:
        yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
    
    if 'logdir' not in yaml_config:
        print('did not find a log dir')
        logdir = f"/logdir-{datetime.now().strftime('%y%m%d-%H%M%S')}" # ckpts and midi will be saved here
        os.makedirs(logdir, exist_ok=True)
        
    train_wrapper(yaml_config, logdir)
    
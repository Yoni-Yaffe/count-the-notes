import os
from datetime import datetime
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from onsets_and_frames.constants import (
    HOP_LENGTH,
    N_MELS,
    DEFAULT_DEVICE,
    MAX_MIDI,
    MIN_MIDI,
    N_KEYS,
)
from onsets_and_frames.transcriber import OnsetsAndFrames, OnsetsNoFrames
from onsets_and_frames.utils import cycle, initialize_logging_system, get_logger
from onsets_and_frames import constants
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
import time
from conversion_maps import constant_conversion_map
import random
import argparse
import torch
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Onsets and Frames model with optional YAML configuration override"
    )

    # Config file
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Directory to store logs and checkpoints",
    )

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--iterations", type=int, default=20000, help="Training iterations per epoch"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Epoch interval for saving model checkpoints",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--learning-rate-decay-steps",
        type=int,
        default=0,
        help="Learning rate decay steps",
    )
    parser.add_argument(
        "--clip-gradient-norm",
        type=float,
        default=3,
        help="Clip gradient norm value (if any)",
    )

    # Dataset & model config
    parser.add_argument(
        "--transcriber-ckpt",
        type=str,
        default=None,
        help="Path to transcriber checkpoint",
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument(
        "--tsv-dir", type=str, default="NoteEM_tsv", help="Path to TSV metadata files"
    )

    parser.add_argument(
        "--labels-dir-path",
        type=str,
        help="Directory to save or load label files",
    )

    parser.add_argument(
        "--data-dir-path",
        type=str,
        default="datasets",
        help="Root path to training datasets directory",
    )

    # Optional features
    parser.add_argument(
        "--pitch-shift", action="store_true", help="Enable pitch shifting"
    )
    parser.add_argument(
        "--pitch-shift-limit",
        type=int,
        default=5,
        help="Maximum semitones to shift during pitch shifting",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--n-weight", type=float, default=2.0, help="Weight for positive onset loss"
    )
    parser.add_argument("--groups", nargs="+", help="Group names for dataset split")
    parser.add_argument(
        "--make-evaluation", action="store_true", help="Keep evaluation file outputs"
    )
    parser.add_argument(
        "--evaluation-list",
        type=str,
        default=None,
        help="List of files to evaluate only",
    )
    parser.add_argument(
        "--save-to-memory", action="store_true", help="Cache audio features in memory"
    )
    parser.add_argument(
        "--smooth-labels", action="store_true", help="Apply smoothing to labels"
    )
    parser.add_argument(
        "--use-onset-mask", action="store_true", help="Apply onset mask during training"
    )
    parser.add_argument(
        "--with-onset-mask",
        action="store_true",
        help="Use onset mask in model loss computation",
    )

    # Pseudo-labels and label update
    parser.add_argument(
        "--pseudo-labels", action="store_true", help="Enable pseudo-label generation"
    )
    parser.add_argument(
        "--counting-window",
        type=int,
        default=1875,
        help="Window size for label counting. The units are in frames each frame is HOP_LENGTH / SAMPLE_RATE seconds. For example for the defaults of HOP_LENGTH=512 and SAMPLE_RATE=16000 the window size is 1000 frames = 1000 * 512 / 16000 seconds = 32 milliseconds. So window size of 1875 is 1 minute.",
    )
    parser.add_argument(
        "--no-best-dist-update",
        dest="best_dist_update",
        action="store_false",
        help="Disable best distance strategy for label updates (enabled by default). In this strategy we only update the labels if the bag of notes distance between the new labels and the ground truth labels is smaller than the distance between the current labels and the ground truth labels. This is used as a regularization method.",
    )
    parser.add_argument(
        "--best-dist-vec",
        action="store_true",
        help="Use vectorized best distance update",
    )
    parser.add_argument(
        "--peak-size", type=int, default=3, help="Peak size for onset peak detection"
    )
    parser.add_argument(
        "--counting-window-hop",
        type=int,
        default=0,
        help="Hop size for counting window with hop size 0. We do no overlap",
    )
    parser.add_argument(
        "--no-save-updated-labels-midis",
        dest="save_updated_labels_midis",
        action="store_false",
        help="Disable saving of updated label MIDI files (enabled by default)",
    )

    # Onset-only model mode
    parser.add_argument(
        "--onset-no-frames-model",
        action="store_true",
        help="Use onset-only model (OnsetsNoFrames) this is more efficient when training only the onset stack. At the end we just take the frame stack from the initial model",
    )

    # Runtime options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 = main process)",
    )

    return parser.parse_args()


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


def train(
    logdir,
    device,
    iterations,
    checkpoint_interval,
    batch_size,
    sequence_length,
    learning_rate,
    learning_rate_decay_steps,
    clip_gradient_norm,
    epochs,
    transcriber_ckpt,
    config: dict,
):
    # Get the train logger (logging system should already be initialized)
    logger = get_logger("train")
    logger.info(f"config -  {config}")
    logger.info("Cuda is available: %s", torch.cuda.is_available())
    logger.info("Device count: %d", torch.cuda.device_count())
    logger.info("Start time: %s", datetime.now())
    logger.info("device %s", device)
    logger.info("device name %s", torch.cuda.get_device_name(device=device))
    # Place holders
    logger.info("HOP LENGTH %d", HOP_LENGTH)
    logger.info("SEQUENCE LENGTH %d", sequence_length)

    onset_precision = None
    onset_recall = None
    pitch_onset_precision = None
    pitch_onset_recall = None
    loss_list = []
    iter_list = []

    if config.get("seed", None) is not None:
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        logger.info("seed is set to %d", seed)

    total_run_1 = time.time()
    logger.info("device %s", device)
    os.makedirs(logdir, exist_ok=True)
    # n_weight = 1 if HOP_LENGTH == 512 else 2
    n_weight = config["n_weight"]
    dataset_name = config["dataset_name"]

    data_dir_path = config.get("data_dir_path", "datasets")

    train_data_path = os.path.join(data_dir_path, dataset_name, "noteEM_audio")

    if config.get("labels_dir_path") is not None:
        labels_path = config["labels_dir_path"]
    else:
        labels_path = os.path.join(logdir, "NoteEm_labels")

    logger.info("Lables path: %s", labels_path)

    os.makedirs(labels_path, exist_ok=True)
    with open(os.path.join(logdir, "score_log.txt"), "a") as fp:
        fp.write(
            f"Parameters:\ndevice: {device}, iterations: {iterations}, checkpoint_interval: {checkpoint_interval},"
            f" batch_size: {batch_size}, sequence_length: {sequence_length}, learning_rate: {learning_rate}, "
            f"learning_rate_decay_steps: {learning_rate_decay_steps}, clip_gradient_norm: {clip_gradient_norm}, "
            f"epochs: {epochs}, transcriber_ckpt: {transcriber_ckpt}, n_weight: {n_weight}\n"
        )

    if config.get("groups") is not None:
        train_groups = config["groups"]
    else:
        train_groups = [dataset_name]

    conversion_map = constant_conversion_map.conversion_map

    instrument_map = None

    dataset = EMDATASET(
        audio_path=train_data_path,
        tsv_path=config["tsv_dir"],
        labels_path=labels_path,
        groups=train_groups,
        sequence_length=sequence_length,
        seed=42,
        device=DEFAULT_DEVICE,
        instrument_map=instrument_map,
        conversion_map=conversion_map,
        pitch_shift=config["pitch_shift"],
        pitch_shift_limit=config.get("pitch_shift_limit", 5),
        keep_eval_files=config.get("make_evaluation", False),
        evaluation_list=config.get("evaluation_list", None),
        only_eval=(iterations == 0),
        save_to_memory=config.get("save_to_memory", False),
        smooth_labels=config.get("smooth_labels", False),
        use_onset_mask=config.get("use_onset_mask", False),
    )
    if iterations > 0:
        logger.info("len dataset %d %d", len(dataset), len(dataset.data))

    #####
    if transcriber_ckpt is None:
        model_complexity = 64
        onset_complexity = 1.5
        # We create a new transcriber with N_KEYS classes for each instrument:
        transcriber = OnsetsAndFrames(
            N_MELS,
            (MAX_MIDI - MIN_MIDI + 1),
            model_complexity,
            onset_complexity=onset_complexity,
            n_instruments=1,
        )
    else:
        transcriber = torch.load(transcriber_ckpt)

    logger.info("HOP LENGTH %d", constants.HOP_LENGTH, HOP_LENGTH)

    if hasattr(transcriber, "onset_stack"):
        set_diff(transcriber.onset_stack, True)
    if hasattr(transcriber, "offset_stack"):
        set_diff(transcriber.offset_stack, False)
    if hasattr(transcriber, "combined_stack"):
        set_diff(transcriber.combined_stack, False)
    if hasattr(transcriber, "velocity_stack"):
        set_diff(transcriber.velocity_stack, False)

    prev_transcriber = None
    if config.get("onset_no_frames_model", False):
        model_complexity = 64
        onset_complexity = 1.5
        transcriber2 = OnsetsNoFrames(
            N_MELS,
            (MAX_MIDI - MIN_MIDI + 1),
            model_complexity,
            onset_complexity=onset_complexity,
            n_instruments=1,
        ).to(device)
        transcriber2.onset_stack.load_state_dict(transcriber.onset_stack.state_dict())
        prev_transcriber = transcriber
        transcriber = transcriber2
    logger.info("transcriber %s", transcriber)
    parallel_transcriber = DataParallel(transcriber)
    parallel_transcriber = parallel_transcriber.to(device)
    optimizer = torch.optim.Adam(
        list(transcriber.parameters()), lr=learning_rate, weight_decay=0
    )
    transcriber.zero_grad()
    optimizer.zero_grad()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()

        POS = 1.1  # Pseudo-label positive threshold (value > 1 means no pseudo label).
        NEG = -0.1  # Pseudo-label negative threshold (value < 0 means no pseudo label).
        if config["pseudo_labels"]:
            POS = 0.5
            NEG = 0.01

        counting_window = config["counting_window"]
        counting_window_length = counting_window
        # counting_window_length = counting_window * SAMPLE_RATE // HOP_LENGTH
        transcriber.eval()
        to_save_dir = (
            os.path.join(logdir, "alignments")
            if config.get("save_updated_labels_midis", False)
            else None
        )
        with torch.no_grad():
            dataset.update_pts_counting(
                parallel_transcriber,
                counting_window_length,
                POS=POS,
                NEG=NEG,
                FRAME_POS=0.5,
                to_save=to_save_dir,
                first=epoch == 1,
                update=True,
                BEST_DIST=config.get("best_dist_update", False),
                peak_size=config.get("peak_size", 3),
                BEST_DIST_VEC=config.get("best_dist_vec", False),
                counting_window_hop=config.get("counting_window_hop", 0),
            )

        num_workers = config.get("num_workers", 0)
        logger.info("num workers: %d", num_workers)
        if num_workers > 0:
            loader = DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=config.get("num_workers", 0),
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                persistent_workers=True,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=True,
            )

        total_loss = []
        curr_loss = []
        transcriber.train()

        onset_total_tp = 0.0
        onset_total_pp = 0.0
        onset_total_p = 0.0

        torch.cuda.empty_cache()
        loader_cycle = cycle(loader)
        time_start = time.time()
        for iteration in tqdm(
            range(1, iterations + 1), desc=f"Train Loop Epoch {epoch}"
        ):
            curr_loader = loader_cycle
            batch = next(curr_loader)
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            optimizer.zero_grad()
            transcription, transcription_losses = transcriber.run_on_batch(
                batch,
                parallel_model=parallel_transcriber,
                positive_weight=n_weight,
                inv_positive_weight=n_weight,
                with_onset_mask=config.get("with_onset_mask", False),
            )

            onset_pred = transcription["onset"].detach() > 0.5
            onset_total_pp += onset_pred
            onset_tp = onset_pred * batch["onset"].detach()
            onset_total_tp += onset_tp
            onset_total_p += batch["onset"].detach()

            onset_recall = (onset_total_tp.sum() / onset_total_p.sum()).item()
            onset_precision = (onset_total_tp.sum() / onset_total_pp.sum()).item()
            pitch_onset_recall = (
                onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()
            ).item()
            pitch_onset_precision = (
                onset_total_tp[..., -N_KEYS:].sum()
                / onset_total_pp[..., -N_KEYS:].sum()
            ).item()

            transcription_loss = transcription_losses["loss/onset"]

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
                transcriber_path = os.path.join(
                    logdir, "transcriber_iteration_{}.pt".format(iteration)
                )
                if config.get("onset_no_frames_model", False):
                    prev_transcriber.onset_stack.load_state_dict(
                        transcriber.onset_stack.state_dict()
                    )
                    torch.save(prev_transcriber, transcriber_path)
                else:
                    torch.save(transcriber, transcriber_path)
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(logdir, "last-optimizer-state.pt"),
                )
                torch.save(
                    {"instrument_mapping": dataset.instruments},
                    os.path.join(logdir, "instrument_mapping.pt"),
                )

            if epochs == 1 and iteration % 1000 == 1:
                score_msg = (
                    f"iteration {iteration:06d} loss: {np.mean(curr_loss):.5f} Onset Precision:  {onset_precision:.3f} "
                    f"Onset Recall {onset_recall:.3f} Pitch Onset Precision:  {pitch_onset_precision:.3f} "
                    f"Pitch Onset Recall  {pitch_onset_recall:.3f}\n"
                )
                logger.info(score_msg)
                loss_list.append(np.mean(curr_loss))
                iter_list.append(iteration)
                curr_loss = []

                onset_total_tp = 0.0
                onset_total_pp = 0.0
                onset_total_p = 0.0
            elif epochs != 1 and iteration % 1000 == 1:
                loss_list.append(np.mean(curr_loss))
                iter_list.append(iteration + iterations * (epoch - 1))
                curr_loss = []
                score_msg = (
                    f"iteration {iteration:06d} loss: {np.mean(curr_loss):.5f} Onset Precision:  {onset_precision:.3f} "
                    f"Onset Recall {onset_recall:.3f} Pitch Onset Precision:  {pitch_onset_precision:.3f} "
                    f"Pitch Onset Recall  {pitch_onset_recall:.3f}\n"
                )
                logger.info(score_msg)

            if epochs == 1 and iteration % 2500 == 0:
                transcriber_path = os.path.join(logdir, "transcriber_ckpt.pt")
                if config.get("onset_no_frames_model", False):
                    prev_transcriber.onset_stack.load_state_dict(
                        transcriber.onset_stack.state_dict()
                    )
                    torch.save(prev_transcriber, transcriber_path)
                else:
                    torch.save(transcriber, transcriber_path)

        time_end = time.time()
        logger.info(
            "epoch %02d loss: %.5f Onset Precision: %.3f Onset Recall %.3f Pitch Onset Precision: %.3f Pitch Onset Recall %.3f time label update: %s",
            epoch,
            sum(total_loss) / len(total_loss),
            onset_precision,
            onset_recall,
            pitch_onset_precision,
            pitch_onset_recall,
            time.strftime('%M:%S', time.gmtime(time_end - time_start))
        )

        save_condition = epoch % checkpoint_interval == 1 or checkpoint_interval == 1
        if save_condition and epochs != 1:
            torch.save(
                transcriber, os.path.join(logdir, "transcriber_{}.pt".format(epoch))
            )
            torch.save(
                optimizer.state_dict(), os.path.join(logdir, "last-optimizer-state.pt")
            )
            torch.save(
                {"instrument_mapping": dataset.instruments},
                os.path.join(logdir, "instrument_mapping.pt"),
            )
        logger.info(score_msg)

    total_run_2 = time.time()
    logger.info(
        "Total Runtime: %s",
        time.strftime('%H:%M:%S', time.gmtime(total_run_2 - total_run_1))
    )

    # keep last optimized state
    torch.save(optimizer.state_dict(), os.path.join(logdir, "last-optimizer-state.pt"))
    torch.save(
        {"instrument_mapping": dataset.instruments},
        os.path.join(logdir, "instrument_mapping.pt"),
    )


def train_from_args(args):
    logdir = args.logdir or f"logs/logdir-{datetime.now().strftime('%y%m%d-%H%M%S')}"
    os.makedirs(logdir, exist_ok=True)
    # Initialize logging system once and get train logger
    _, _ = initialize_logging_system(logdir)  # Initialize the system
    logger = get_logger("train")  # Get the train logger

    config = vars(args)  # Convert Namespace to dict for easy access
    transcriber_ckpt = args.transcriber_ckpt
    checkpoint_interval = args.checkpoint_interval
    batch_size = args.batch_size
    iterations = args.iterations
    learning_rate = args.learning_rate
    learning_rate_decay_steps = args.learning_rate_decay_steps
    clip_gradient_norm = args.clip_gradient_norm
    epochs = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequence_length = constants.SEQ_LEN

    logger.info(f"Log directory: {logdir}")

    try:
        config_path = os.path.join(logdir, "args_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Saved config to {config_path}")
    except Exception as e:
        logger.warning("Failed to save command-line arguments to JSON.")
        logger.error("Error: %s", str(e))

    train(
        logdir,
        device,
        iterations,
        checkpoint_interval,
        batch_size,
        sequence_length,
        learning_rate,
        learning_rate_decay_steps,
        clip_gradient_norm,
        epochs,
        transcriber_ckpt,
        config,
    )


if __name__ == "__main__":
    args = parse_args()
    train_from_args(args)

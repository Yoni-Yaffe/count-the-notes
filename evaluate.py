import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import (
    precision_recall_f1_overlap as evaluate_notes_with_velocity,
)
from scipy.stats import hmean
from tqdm import tqdm

from onsets_and_frames.constants import (
    HOP_LENGTH,
    MAX_MIDI,
    MIN_MIDI,
    SAMPLE_RATE,
)
from onsets_and_frames.decoding import notes_to_frames
from onsets_and_frames.midi_utils import parse_midi_multi

eps = sys.float_info.epsilon


def midi2hz(midi):
    res = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    return res


def evaluate_single_pair(args):
    """
    Evaluates a single pair of synthesized and reference MIDI files.
    """
    transcribed, reference, tolerance, shift = args
    metrics = defaultdict(list)

    try:
        if reference.endswith(".tsv"):
            reference_events = np.loadtxt(reference, delimiter="\t", skiprows=1)
            reference_events[:, :2] = reference_events[:, :2] + shift
            reference_events[:, 1] = reference_events[:, 1] + 0.0001
            print("shift", shift)
        else:
            reference_events = parse_midi_multi(reference)
        transcribed_events = parse_midi_multi(transcribed)

        max_time = int(reference_events[:, 1].max() + 5)
        audio_length = max_time * SAMPLE_RATE
        n_keys = MAX_MIDI - MIN_MIDI + 1

        n_steps = (audio_length - 1) // HOP_LENGTH + 1
        n_steps_transcriber = (audio_length - 1) // HOP_LENGTH + 1

        p_est, i_est, v_est, ins_est = (
            transcribed_events[:, 2],
            transcribed_events[:, 0:2],
            transcribed_events[:, 3],
            transcribed_events[:, 4],
        )

        p_ref, i_ref, v_ref, ins_ref = (
            reference_events[:, 2],
            reference_events[:, 0:2],
            reference_events[:, 3],
            reference_events[:, 4],
        )
        print("Available instruments", np.unique(ins_ref))
        p_ref_2 = np.array([int(midi - MIN_MIDI) for midi in p_ref])
        i_ref_2 = np.array(
            [
                (
                    int(round(on * SAMPLE_RATE / HOP_LENGTH)),
                    int(round(off * SAMPLE_RATE / HOP_LENGTH)),
                )
                for on, off in i_ref
            ]
        )
        p_est_2 = np.array([int(midi - MIN_MIDI) for midi in p_est])
        i_est_2 = np.array(
            [
                (
                    int(round(on * SAMPLE_RATE / HOP_LENGTH)),
                    int(round(off * SAMPLE_RATE / HOP_LENGTH)),
                )
                for on, off in i_est
            ]
        )

        t_ref, f_ref = notes_to_frames(p_ref_2, i_ref_2, (n_steps, n_keys))
        t_est, f_est = notes_to_frames(p_est_2, i_est_2, (n_steps_transcriber, n_keys))

        scaling = HOP_LENGTH / SAMPLE_RATE

        p_ref = np.array([midi2hz(midi) for midi in p_ref])
        p_est = np.array([midi2hz(midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [
            np.array([midi2hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref
        ]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [
            np.array([midi2hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est
        ]

        p, r, f, o = evaluate_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance
        )
        print("onset:", p, r, f, o)
        on_p, on_r, on_f = p, r, f

        metrics["metric/note/precision"].append(p)
        metrics["metric/note/recall"].append(r)
        metrics["metric/note/f1"].append(f)
        metrics["metric/note/overlap"].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        print("onset-offset:", p, r, f, o)

        metrics["metric/note-with-offsets/precision"].append(p)
        metrics["metric/note-with-offsets/recall"].append(r)
        metrics["metric/note-with-offsets/f1"].append(f)
        metrics["metric/note-with-offsets/overlap"].append(o)

        p, r, f, o = evaluate_notes_with_velocity(
            i_ref,
            p_ref,
            v_ref,
            i_est,
            p_est,
            v_est,
            velocity_tolerance=0.1,
            offset_ratio=None,
            onset_tolerance=tolerance,
        )
        print("onset-velocity:", p, r, f, o)
        metrics["metric/note-with-velocity/precision"].append(p)
        metrics["metric/note-with-velocity/recall"].append(r)
        metrics["metric/note-with-velocity/f1"].append(f)
        metrics["metric/note-with-velocity/overlap"].append(o)

        p, r, f, o = evaluate_notes_with_velocity(
            i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1
        )
        print("onset-offset-velocity:", p, r, f, o)
        metrics["metric/note-with-offsets-and-velocity/precision"].append(p)
        metrics["metric/note-with-offsets-and-velocity/recall"].append(r)
        metrics["metric/note-with-offsets-and-velocity/f1"].append(f)
        metrics["metric/note-with-offsets-and-velocity/overlap"].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        fr_p, fr_r, fr_f = (
            frame_metrics["Precision"],
            frame_metrics["Recall"],
            hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps])
            - eps,
        )
        metrics["metric/frame/f1"].append(
            hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps])
            - eps
        )
        print(
            "frame p/r/f:",
            frame_metrics["Precision"],
            frame_metrics["Recall"],
            hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps])
            - eps,
        )

        for key, loss in frame_metrics.items():
            metrics["metric/frame/" + key.lower().replace(" ", "_")].append(loss)

        torch.cuda.empty_cache()

        # Add more evaluations as needed...
    except Exception as e:
        print(f"Error evaluating pair {transcribed} and {reference}: {e}")

    metrics["file_names"] = [os.path.basename(transcribed), os.path.basename(reference)]
    return metrics


def evaluate(
    synthsized_midis,
    reference_midis,
    tolerance=0.05,
    shift=0.0,
    outfile=None,
):
    """
    Serial evaluation of MIDI file pairs.
    """
    args_list = [
        (transcribed, reference, tolerance, shift)
        for transcribed, reference in zip(synthsized_midis, reference_midis)
    ]
    print(f"tolerance {tolerance} shift {shift}")
    print("number of files", len(args_list))

    per_piece_data = []
    combined_metrics = defaultdict(list)

    for args in tqdm(args_list, desc="Evaluating"):
        metrics = evaluate_single_pair(args)
        if "file_names" in metrics:
            file_names = metrics.pop("file_names")
            row = {
                "Transcribed File": file_names[0],
                "Reference File": file_names[1],
            }
            row.update({key: np.mean(values) for key, values in metrics.items()})
            per_piece_data.append(row)

        for key, values in metrics.items():
            combined_metrics[key].extend(values)

    for key, values in combined_metrics.items():
        if key.startswith("metric/"):
            _, category, name = key.split("/")
            print(
                f"{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}"
            )

    if outfile:
        with open(outfile, "w") as fp:
            for key, values in combined_metrics.items():
                if key.startswith("metric/"):
                    _, category, name = key.split("/")
                    fp.write(
                        f"{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}\n"
                    )

        df = pd.DataFrame(per_piece_data)
        csv_file = outfile.replace(".txt", "_per_piece.csv")
        df.to_csv(csv_file, index=False)

    return combined_metrics


def evaluate_parallel(
    synthsized_midis,
    reference_midis,
    tolerance=0.05,
    max_workers=4,
    outfile=None,
    shift=0.0,
):
    """
    Parallel evaluation of MIDI file pairs.
    """
    print("The number of available cores is", os.cpu_count())
    args_list = [
        (transcribed, reference, tolerance, shift)
        for transcribed, reference in zip(synthsized_midis, reference_midis)
    ]
    print(f"tolerance {tolerance} shift {shift}")
    print("number of files", len(args_list))
    per_piece_data = []

    combined_metrics = defaultdict(list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_pair, args): args for args in args_list
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating"
        ):
            metrics = future.result()
            if "file_names" in metrics:
                file_names = metrics.pop("file_names")
                row = {
                    "Transcribed File": file_names[0],
                    "Reference File": file_names[1],
                }
                row.update({key: np.mean(values) for key, values in metrics.items()})
                per_piece_data.append(row)

            for key, values in metrics.items():
                combined_metrics[key].extend(values)

    for key, values in combined_metrics.items():
        if key.startswith("metric/"):
            _, category, name = key.split("/")
            print(
                f"{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}"
            )
    if outfile:
        with open(outfile, "w") as fp:
            for key, values in combined_metrics.items():
                if key.startswith("metric/"):
                    _, category, name = key.split("/")
                    fp.write(
                        f"{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}\n"
                    )

        df = pd.DataFrame(per_piece_data)
        csv_file = outfile.replace(".txt", "_per_piece.csv")
        df.to_csv(csv_file, index=False)

    return combined_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate transcribed MIDI files against reference MIDI files using note- and frame-level metrics."
    )

    parser.add_argument(
        "--transcribed-dir",
        type=str,
        required=True,
        help="Directory containing transcribed MIDI files.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        required=True,
        help="Directory containing reference MIDI files or TSV files. The names should match the transcribed files except the extension.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel evaluation using multiprocessing.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Path to output summary text file (and CSV per-piece scores).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Onset tolerance in seconds (default: 0.05).",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="Shift reference MIDI onset/offset times (in seconds).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of workers for parallel processing (default: 4).",
    )

    args = parser.parse_args()

    transcribed_midis = [
        os.path.join(args.transcribed_dir, f)
        for f in sorted(os.listdir(args.transcribed_dir))
        if f.lower().endswith((".mid", ".midi"))
    ]
    reference_midis = [
        os.path.join(args.reference_dir, f)
        for f in sorted(os.listdir(args.reference_dir))
        if f.lower().endswith((".mid", ".midi", ".tsv"))
    ]

    if len(transcribed_midis) != len(reference_midis):
        raise ValueError(
            f"The number of transcribed MIDI files does not match the number of reference MIDI/TSV files. Number of transcribed files: {len(transcribed_midis)}, Number of reference files: {len(reference_midis)}."
        )

    if args.parallel:
        evaluate_parallel(
            transcribed_midis,
            reference_midis,
            tolerance=args.tolerance,
            max_workers=args.max_workers,
            outfile=args.outfile,
            shift=args.shift,
        )
    else:
        evaluate(
            transcribed_midis,
            reference_midis,
            tolerance=args.tolerance,
            shift=args.shift,
            outfile=args.outfile,
        )

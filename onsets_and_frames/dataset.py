import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import os
from onsets_and_frames import constants
import torch
from onsets_and_frames.mel import melspectrogram
from onsets_and_frames.midi_utils import (
    midi_to_frames,
    save_midi_alignments_and_predictions,
)
from onsets_and_frames.utils import (
    smooth_labels,
    shift_label,
    get_diff,
    get_peaks,
    get_logger,
)
from onsets_and_frames.constants import N_KEYS, SAMPLE_RATE, DEFAULT_DEVICE
import time
import sys
import librosa


class EMDATASET(Dataset):
    def __init__(
        self,
        audio_path="NoteEM_audio",
        tsv_path="NoteEM_tsv",
        labels_path="NoteEm_labels",
        groups=None,
        sequence_length=None,
        seed=42,
        device=DEFAULT_DEVICE,
        instrument_map=None,
        update_instruments=False,
        transcriber=None,
        conversion_map=None,
        pitch_shift=True,
        pitch_shift_limit=5,
        keep_eval_files=False,
        n_eval=1,
        evaluation_list=None,
        only_eval=False,
        save_to_memory=False,
        smooth_labels=False,
        use_onset_mask=False,
    ):
        # Get the dataset logger (logging system should already be initialized by train.py)
        self.logger = get_logger("dataset")

        self.audio_path = audio_path
        self.tsv_path = tsv_path
        self.labels_path = labels_path
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.groups = groups
        self.conversion_map = conversion_map
        self.eval_file_list = []
        self.file_list = self.files(
            self.groups,
            pitch_shift=pitch_shift,
            keep_eval_files=keep_eval_files,
            n_eval=n_eval,
            evaluation_list=evaluation_list,
            pitch_shift_limit=pitch_shift_limit,
        )
        self.save_to_memory = save_to_memory
        self.smooth_labels = smooth_labels
        self.use_onset_mask = use_onset_mask
        self.pitch_shift_limit = pitch_shift_limit

        self.logger.debug("Save to memory is %s", self.save_to_memory)
        self.logger.info("len file list %d", len(self.file_list))
        self.logger.info("\n\n")

        if instrument_map is None:
            self.get_instruments(conversion_map=conversion_map)
        else:
            self.instruments = instrument_map
            if update_instruments:
                self.add_instruments()
        self.transcriber = transcriber
        if only_eval:
            return
        self.load_pts(self.file_list)
        self.data = []
        self.logger.info("Reading files...")
        for input_files in tqdm(self.file_list, desc="creating data list"):
            flac, _ = input_files
            audio_len = librosa.get_duration(path=flac)
            minutes = int(np.ceil(audio_len / 60))
            copies = minutes
            for _ in range(copies):
                self.data.append(input_files)
        random.shuffle(self.data)

    def flac_to_pt_path(self, flac):
        pt_fname = os.path.basename(flac).replace(".flac", ".pt")
        pt_path = os.path.join(self.labels_path, pt_fname)
        return pt_path

    def __len__(self):
        return len(self.data)

    def files(
        self,
        groups,
        pitch_shift=True,
        keep_eval_files=False,
        n_eval=1,
        evaluation_list=None,
        pitch_shift_limit=5,
    ):
        self.path = self.audio_path
        tsvs_path = self.tsv_path
        self.logger.info("tsv path: %s", tsvs_path)
        self.logger.info("Evaluation list: %s", evaluation_list)
        res = []
        self.logger.info("keep eval files: %s", keep_eval_files)
        self.logger.info("n eval: %d", n_eval)
        for group in groups:
            tsvs = os.listdir(tsvs_path + os.sep + group)
            tsvs = sorted(tsvs)
            if keep_eval_files and evaluation_list is None:
                eval_tsvs = tsvs[:n_eval]
                tsvs = tsvs[n_eval:]
            elif keep_eval_files and evaluation_list is not None:
                eval_tsvs_names = [
                    i.split("#")[0].split(".flac")[0].split(".tsv")[0]
                    for i in evaluation_list
                ]
                eval_tsvs = [
                    i
                    for i in tsvs
                    if i.split("#")[0].split(".tsv")[0] in eval_tsvs_names
                ]
                tsvs = [i for i in tsvs if i not in eval_tsvs]
            else:
                eval_tsvs = []
            self.logger.info("len tsvs: %d", len(tsvs))

            tsvs_names = [t.split(".tsv")[0].split("#")[0] for t in tsvs]
            eval_tsvs_names = [t.split(".tsv")[0].split("#")[0] for t in eval_tsvs]
            for shft in range(-5, 6):
                if shft != 0 and not pitch_shift or abs(shft) > pitch_shift_limit:
                    continue
                curr_fls_pth = self.path + os.sep + group + "#{}".format(shft)

                fls = os.listdir(curr_fls_pth)
                orig_files = fls
                # print(f"files names before\n {fls}")
                fls = [
                    i for i in fls if i.split("#")[0] in tsvs_names
                ]  # in case we dont have the corresponding midi
                missing_fls = [i for i in orig_files if i not in fls]
                if len(missing_fls) > 0:
                    self.logger.warning("missing files: %s", missing_fls)
                fls_names = [i.split("#")[0].split(".flac")[0] for i in fls]
                tsvs = [
                    i for i in tsvs if i.split(".tsv")[0].split("#")[0] in fls_names
                ]
                assert len(tsvs) == len(fls)
                # print(f"files names after\n {fls}")
                fls = sorted(fls)

                if shft == 0:
                    eval_fls = os.listdir(curr_fls_pth)
                    # print(f"files names\n {eval_fls}")
                    eval_fls = [
                        i for i in eval_fls if i.split("#")[0] in eval_tsvs_names
                    ]  # in case we dont have the corresponding midi
                    eval_fls_names = [i.split("#")[0] for i in eval_fls]
                    eval_tsvs = [
                        i
                        for i in eval_tsvs
                        if i.split(".tsv")[0].split("#")[0] in eval_fls_names
                    ]
                    assert len(eval_fls_names) == len(eval_tsvs_names)
                    # print(f"files names\n {eval_fls}")
                    eval_fls = sorted(eval_fls)
                    for f, t in zip(eval_fls, eval_tsvs):
                        self.eval_file_list.append(
                            (
                                curr_fls_pth + os.sep + f,
                                tsvs_path + os.sep + group + os.sep + t,
                            )
                        )

                for f, t in zip(fls, tsvs):
                    res.append(
                        (
                            curr_fls_pth + os.sep + f,
                            tsvs_path + os.sep + group + os.sep + t,
                        )
                    )

        for flac, tsv in res:
            if (
                os.path.basename(flac).split("#")[0].split(".flac")[0]
                != os.path.basename(tsv).split("#")[0].split(".tsv")[0]
            ):
                self.logger.warning("found mismatch in the files: ")
                self.logger.warning("flac: %s", os.path.basename(flac).split("#")[0])
                self.logger.warning("tsv: %s", os.path.basename(tsv).split("#")[0])
                self.logger.warning("please check the input files")
                exit(1)
        return res

    def get_instruments(self, conversion_map=None):
        instruments = set()
        for _, f in self.file_list:
            events = np.loadtxt(f, delimiter="\t", skiprows=1)
            curr_instruments = set(events[:, -1])
            if conversion_map is not None:
                curr_instruments = {
                    conversion_map[c] if c in conversion_map else c
                    for c in curr_instruments
                }
            instruments = instruments.union(curr_instruments)
        instruments = [int(elem) for elem in instruments if elem < 115]
        if conversion_map is not None:
            instruments = [i for i in instruments if i in conversion_map]
        instruments = list(set(instruments))
        if 0 in instruments:
            piano_ind = instruments.index(0)
            instruments.pop(piano_ind)
            instruments.insert(0, 0)
        self.instruments = instruments
        self.instruments = list(
            set(self.instruments) - set(range(88, 104)) - set(range(112, 150))
        )
        self.logger.info("Dataset instruments: %s", self.instruments)
        self.logger.info("Total: %d instruments", len(self.instruments))

    def add_instruments(self):
        for _, f in self.file_list:
            events = np.loadtxt(f, delimiter="\t", skiprows=1)
            curr_instruments = set(events[:, -1])
            new_instruments = curr_instruments - set(self.instruments)
            self.instruments += list(new_instruments)
        instruments = [int(elem) for elem in self.instruments if (elem < 115)]
        self.instruments = instruments

    def __getitem__(self, index):
        data = self.load(*self.data[index])
        # result = dict(path=data['path'])
        midi_length = len(data["label"])
        n_steps = self.sequence_length // constants.HOP_LENGTH
        if midi_length < n_steps:
            step_begin = 0
            step_end = midi_length
        else:
            step_begin = self.random.randint(max(midi_length - n_steps, 1))
            step_end = step_begin + n_steps
        begin = step_begin * constants.HOP_LENGTH
        end = begin + self.sequence_length

        audio = (
            data["audio"][begin:end].float().div_(32768.0)
        )  # torch.ShortTensor → float
        label = data["label"][step_begin:step_end].clone()  # torch.Tensor

        if audio.shape[0] < self.sequence_length:
            pad_amt = self.sequence_length - audio.shape[0]
            audio = torch.cat([audio, torch.zeros(pad_amt, dtype=audio.dtype)], dim=0)

        if label.shape[0] < n_steps:
            pad_amt = n_steps - label.shape[0]
            label = torch.cat(
                [label, torch.zeros((pad_amt, *label.shape[1:]), dtype=label.dtype)],
                dim=0,
            )

        audio = torch.clamp(audio, -1.0, 1.0)
        result = {"path": data["path"], "audio": audio, "label": label}
        if "velocity" in data:
            result["velocity"] = data["velocity"][step_begin:step_end, ...]
            result["velocity"] = result["velocity"].float() / 128.0

        if result["label"].max() < 3:
            result["onset"] = result["label"].float()
        else:
            result["onset"] = (result["label"] == 3).float()

        result["offset"] = (result["label"] == 1).float()
        result["frame"] = (result["label"] > 1).float()

        if self.smooth_labels:
            result["onset"] = smooth_labels(result["onset"])
        if self.use_onset_mask:
            if "onset_mask" in data:
                result["onset_mask"] = data["onset_mask"][
                    step_begin:step_end, ...
                ].float()
            else:
                result["onset_mask"] = torch.ones_like(result["onset"]).float()
            if "frame_mask" in data:
                result["frame_mask"] = data["frame_mask"][
                    step_begin:step_end, ...
                ].float()
            else:
                result["frame_mask"] = torch.ones_like(result["frame"]).float()

        shape = result["frame"].shape
        keys = N_KEYS
        new_shape = shape[:-1] + (shape[-1] // keys, keys)
        result["big_frame"] = result["frame"]
        result["frame"], _ = result["frame"].reshape(new_shape).max(axis=-2)

        # if 'frame_mask' not in data:
        #     result['frame_mask'] = torch.ones_like(result['frame']).to(self.device).float()

        result["big_offset"] = result["offset"]
        result["offset"], _ = result["offset"].reshape(new_shape).max(axis=-2)
        result["group"] = self.data[index][0].split(os.sep)[-2].split("#")[0]

        return result

    def load(self, audio_path, tsv_path):
        if self.save_to_memory:
            data = self.pts[audio_path]
        else:
            data = torch.load(self.flac_to_pt_path(audio_path))
        if len(data["audio"].shape) > 1:
            data["audio"] = (data["audio"].float().mean(dim=-1)).short()
        if "label" in data:
            return data
        else:
            piece, part = audio_path.split(os.sep)[-2:]
            piece_split = piece.split("#")
            if len(piece_split) == 2:
                piece, shift1 = piece_split
            else:
                piece, shift1 = "#".join(piece_split[:2]), piece_split[-1]
            part_split = part.split("#")
            if len(part_split) == 2:
                part, shift2 = part_split
            else:
                part, shift2 = "#".join(part_split[:2]), part_split[-1]
            shift2, _ = shift2.split(".")
            assert shift1 == shift2
            shift = shift1
            assert shift != 0
            orig = audio_path.replace("#{}".format(shift), "#0")
            if self.save_to_memory:
                orig_data = self.pts[orig]
            else:
                orig_data = torch.load(self.flac_to_pt_path(orig))
            res = {}
            res["label"] = shift_label(orig_data["label"], int(shift))
            res["path"] = audio_path
            res["audio"] = data["audio"]
            if "velocity" in orig_data:
                res["velocity"] = shift_label(orig_data["velocity"], int(shift))
            if "onset_mask" in orig_data:
                res["onset_mask"] = shift_label(orig_data["onset_mask"], int(shift))
            if "frame_mask" in orig_data:
                res["frame_mask"] = shift_label(orig_data["frame_mask"], int(shift))
            return res

    def load_pts(self, files):
        self.pts = {}
        self.logger.info("loading pts...")
        for flac, tsv in tqdm(files, desc="loading pts"):
            # print('flac, tsv', flac, tsv)
            if os.path.isfile(
                self.labels_path
                + os.sep
                + flac.split(os.sep)[-1].replace(".flac", ".pt")
            ):
                if self.save_to_memory:
                    self.pts[flac] = torch.load(
                        self.labels_path
                        + os.sep
                        + flac.split(os.sep)[-1].replace(".flac", ".pt")
                    )
            else:
                if flac.count("#") != 2:
                    self.logger.debug("two # in filename: %s", flac)
                audio, sr = soundfile.read(flac, dtype="int16")
                if len(audio.shape) == 2:
                    audio = audio.astype(float).mean(axis=1)
                else:
                    audio = audio.astype(float)
                audio = audio.astype(np.int16)
                self.logger.debug("audio len: %d", len(audio))
                assert sr == SAMPLE_RATE
                audio = torch.ShortTensor(audio)
                if "#0" not in flac:
                    assert "#" in flac
                    data = {"audio": audio}
                    if self.save_to_memory:
                        self.pts[flac] = data
                    torch.save(data, self.flac_to_pt_path(flac))
                    continue
                midi = np.loadtxt(tsv, delimiter="\t", skiprows=1)
                unaligned_label = midi_to_frames(
                    midi, self.instruments, conversion_map=self.conversion_map
                )
                if len(self.instruments) == 1:
                    unaligned_label = unaligned_label[:, -N_KEYS:]
                if len(unaligned_label) < self.sequence_length // constants.HOP_LENGTH:
                    diff = self.sequence_length // constants.HOP_LENGTH - len(
                        unaligned_label
                    )
                    pad = torch.zeros(
                        (diff, unaligned_label.shape[1]), dtype=unaligned_label.dtype
                    )
                    unaligned_label = torch.cat((unaligned_label, pad), dim=0)

                group = flac.split(os.sep)[-2].split("#")[0]
                data = dict(
                    path=self.labels_path + os.sep + flac.split(os.sep)[-1],
                    audio=audio,
                    unaligned_label=unaligned_label,
                    group=group,
                    BON=float("inf"),
                    BON_VEC=np.full(unaligned_label.shape[1], float("inf")),
                )

                torch.save(data, self.flac_to_pt_path(flac))
                if self.save_to_memory:
                    self.pts[flac] = data

    def update_pts_counting(
        self,
        transcriber,
        counting_window_length,
        POS=1.1,
        NEG=-0.001,
        FRAME_POS=0.5,
        to_save=None,
        first=False,
        update=True,
        BEST_DIST=False,
        peak_size=3,
        BEST_DIST_VEC=False,
        counting_window_hop=0,
    ):
        self.logger.info("Updating pts...")
        self.logger.info("First %s", first)
        total_counting_time = 0.0  # Initialize total time for counting-based alignment

        self.logger.info("POS, NEG: %s, %s", POS, NEG)
        if to_save is not None:
            os.makedirs(to_save, exist_ok=True)
        self.logger.info("There are %d pts", len(self.pts))
        update_count = 0
        sys.stdout.flush()
        onlt_pitch_0_files = [f for f in self.file_list if "#0" in f[0]]
        for input_files in tqdm(onlt_pitch_0_files, desc="updating pts"):
            flac, tsv = input_files
            data = torch.load(self.flac_to_pt_path(flac))
            if "unaligned_label" not in data:
                self.logger.warning("No unaligned labels for %s", flac)
                continue
            audio_inp = data["audio"].float() / 32768.0
            MAX_TIME = 5 * 60 * SAMPLE_RATE
            audio_inp_len = len(audio_inp)
            if audio_inp_len > MAX_TIME:
                n_segments = int(np.ceil(audio_inp_len / MAX_TIME))
                self.logger.info("Long audio, splitting to %d segments", n_segments)
                seg_len = MAX_TIME
                onsets_preds = []
                offset_preds = []
                frame_preds = []
                for i_s in range(n_segments):
                    curr = (
                        audio_inp[i_s * seg_len : (i_s + 1) * seg_len]
                        .unsqueeze(0)
                        .cuda()
                    )
                    curr_mel = melspectrogram(
                        curr.reshape(-1, curr.shape[-1])[:, :-1]
                    ).transpose(-1, -2)
                    (
                        curr_onset_pred,
                        curr_offset_pred,
                        _,
                        curr_frame_pred,
                        curr_velocity_pred,
                    ) = transcriber(curr_mel)
                    onsets_preds.append(curr_onset_pred)
                    offset_preds.append(curr_offset_pred)
                    frame_preds.append(curr_frame_pred)
                onset_pred = torch.cat(onsets_preds, dim=1)
                offset_pred = torch.cat(offset_preds, dim=1)
                frame_pred = torch.cat(frame_preds, dim=1)
            else:
                audio_inp = audio_inp.unsqueeze(0).cuda()
                mel = melspectrogram(
                    audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]
                ).transpose(-1, -2)
                onset_pred, offset_pred, _, frame_pred, _ = transcriber(mel)
            self.logger.debug("Done predicting.")

            # We assume onset predictions are of length N_KEYS * (len(instruments) + 1),
            # first N_KEYS classes are the first instrument, next N_KEYS classes are the next instrument, etc.,
            # and last N_KEYS classes are for pitch regardless of instrument
            # Currently, frame and offset predictions are only N_KEYS classes.
            onset_pred = onset_pred.detach().squeeze().cpu()
            frame_pred = frame_pred.detach().squeeze().cpu()

            PEAK_SIZE = peak_size
            self.logger.debug("PEAK_SIZE: %d", PEAK_SIZE)
            # we peak peak the onset prediction to only keep local maximum onsets
            if peak_size > 0:
                peaks = get_peaks(
                    onset_pred, PEAK_SIZE
                )  # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
                onset_pred[~peaks] = 0

            unaligned_onsets = (data["unaligned_label"] == 3).float().numpy()

            onset_pred_np = onset_pred.numpy()
            frame_pred_np = frame_pred.numpy()

            pred_bag_of_notes = (onset_pred_np[:, -N_KEYS:] >= 0.5).sum(axis=0)
            gt_bag_of_notes = unaligned_onsets[:, -N_KEYS:].astype(bool).sum(axis=0)
            bon_dist = (((pred_bag_of_notes - gt_bag_of_notes) ** 2).sum()) ** 0.5

            pred_bag_of_notes_with_inst = (onset_pred_np >= 0.5).sum(axis=0)
            gt_bag_of_notes_with_inst = unaligned_onsets.astype(bool).sum(axis=0)
            bon_dist_vec = np.abs(
                pred_bag_of_notes_with_inst - gt_bag_of_notes_with_inst
            )

            bon_dist /= gt_bag_of_notes.sum()
            self.logger.debug("bag of notes dist: %f", bon_dist)
            ####

            aligned_onsets = np.zeros(onset_pred_np.shape, dtype=bool)
            aligned_frames = np.zeros(onset_pred_np.shape, dtype=bool)

            # This block is the main difference between the counting approach and the DTW approach.
            # In the counting approach we label the audio by counting note onsets: For each onset pitch class,
            # denote by K the number of times it occurs in the unaligned label. We simply take the K highest local
            # peaks predicted by the current model.
            # Split unaligned onsets into chunks of size counting_window_length
            self.logger.debug(
                "unaligned onsets shape: %s, counting window length: %d, counting window hop: %d",
                unaligned_onsets.shape,
                counting_window_length,
                counting_window_hop,
            )
            assert counting_window_hop <= counting_window_length
            if counting_window_hop == 0:
                counting_window_hop = counting_window_length

            num_chunks = (
                1
                if counting_window_length == 0
                else int(np.ceil(len(unaligned_onsets) / counting_window_hop))
            )

            self.logger.debug("number of chunks: %d", num_chunks)
            start_time = time.time()
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * counting_window_hop
                if counting_window_length == 0:
                    end_idx = max(len(unaligned_onsets), len(onset_pred_np))
                else:
                    end_idx = min(
                        start_idx + counting_window_length, len(unaligned_onsets)
                    )
                chunk_onsets = unaligned_onsets[start_idx:end_idx]
                chunk_onsets_count = (
                    (data["unaligned_label"][start_idx:end_idx, :] == 3)
                    .sum(dim=0)
                    .numpy()
                )

                for f, f_count in enumerate(chunk_onsets_count):
                    if f_count == 0:
                        continue
                    f_most_likely = np.sort(
                        onset_pred_np[start_idx:end_idx, f].argsort()[::-1][:f_count]
                    )
                    f_most_likely += start_idx  # Adjust indices to the original size
                    aligned_onsets[f_most_likely, f] = 1

                    f_unaligned = chunk_onsets[:, f].nonzero()
                    assert len(f_unaligned) == 1
                    f_unaligned = f_unaligned[0]

            counting_duration = time.time() - start_time
            total_counting_time += counting_duration
            self.logger.debug(
                "Counting alignment for file '%s' took %.2f seconds.",
                flac,
                counting_duration,
            )

            # Pseudo labels, Pos bigger than 1 is equivalent to not using pseudo labels
            pseudo_onsets = (onset_pred_np >= POS) & (~aligned_onsets)

            onset_label = np.maximum(pseudo_onsets, aligned_onsets)

            # in this project we do not train frame stack but we calculate the labeels anyways
            pseudo_frames = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            pseudo_offsets = np.zeros(pseudo_onsets.shape, dtype=pseudo_onsets.dtype)
            for t, f in zip(*onset_label.nonzero()):
                t_off = t
                while (
                    t_off < len(pseudo_frames)
                    and frame_pred[t_off, f % N_KEYS] >= FRAME_POS
                ):
                    t_off += 1
                pseudo_frames[t:t_off, f] = 1
                if t_off < len(pseudo_offsets):
                    pseudo_offsets[t_off, f] = 1
            frame_label = np.maximum(pseudo_frames, aligned_frames)
            offset_label = get_diff(frame_label, offset=True)

            label = np.maximum(2 * frame_label, offset_label)
            label = np.maximum(3 * onset_label, label).astype(np.uint8)

            if to_save is not None:
                save_midi_alignments_and_predictions(
                    to_save,
                    data["path"],
                    self.instruments,
                    aligned_onsets,
                    aligned_frames,
                    onset_pred_np,
                    frame_pred_np,
                    prefix="",
                    group=data["group"],
                )
            prev_bon_dist = data.get("BON", float("inf"))
            prev_bon_dist_vec = data.get("BON_VEC", None)
            if update:
                if BEST_DIST_VEC:
                    self.logger.debug("Updated Labels")
                    if prev_bon_dist_vec is None:
                        raise ValueError(
                            "BEST_DIST_VEC is True but no previous BON_VEC found"
                        )
                    prev_label = data["label"]
                    new_label = torch.from_numpy(label).byte()
                    if first:
                        prev_label = new_label
                        update_count += 1
                    else:
                        updated_flag = False
                        num_pitches_updated = 0
                        for k in range(prev_label.shape[1]):
                            if prev_bon_dist_vec[k] > bon_dist_vec[k]:
                                prev_label[:, k] = new_label[:, k]
                                prev_bon_dist_vec[k] = bon_dist_vec[k]
                                num_pitches_updated += 1
                                updated_flag = True
                        if updated_flag:
                            update_count += 1
                        self.logger.debug("Updated %d pitches", num_pitches_updated)
                    data["label"] = prev_label
                    data["BON_VEC"] = prev_bon_dist_vec
                    self.logger.debug("saved updated pt")
                    torch.save(
                        data,
                        self.labels_path
                        + os.sep
                        + flac.split(os.sep)[-1]
                        .replace(".flac", ".pt")
                        .replace(".mp3", ".pt"),
                    )

                elif not BEST_DIST or bon_dist < prev_bon_dist:
                    update_count += 1
                    self.logger.debug("Updated Labels")

                    data["label"] = torch.from_numpy(label).byte()

                    data["BON"] = bon_dist
                    self.logger.debug("saved updated pt")
                    torch.save(
                        data,
                        self.labels_path
                        + os.sep
                        + flac.split(os.sep)[-1]
                        .replace(".flac", ".pt")
                        .replace(".mp3", ".pt"),
                    )

            if bon_dist < prev_bon_dist:
                self.logger.debug(
                    "Bag of notes distance improved from %f to %f",
                    prev_bon_dist,
                    bon_dist,
                )
                data["BON"] = bon_dist

                if to_save is not None and BEST_DIST:
                    os.makedirs(to_save + "/BEST_BON", exist_ok=True)
                    save_midi_alignments_and_predictions(
                        to_save + "/BEST_BON",
                        data["path"],
                        self.instruments,
                        aligned_onsets,
                        aligned_frames,
                        onset_pred_np,
                        frame_pred_np,
                        prefix="BEST_BON",
                        group=data["group"],
                        use_time=False,
                    )

        self.logger.info(
            "Updated %d pts out of %d", update_count, len(onlt_pitch_0_files)
        )
        self.logger.info(
            "Total counting alignment time for all files: %.2f seconds.", total_counting_time
        )

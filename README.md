# CountEM — *Count the Notes*

## Histogram‑based supervision for Automatic Music Transcription (AMT) ISMIR 2025 

**ISMIR 2025 paper codebase**

CountEM is a training framework for automatic music transcription (AMT) that relies only on unordered note count histograms, eliminating the need for aligned or ordered labels.
This repository provides the official implementation of our ISMIR 2025 paper:
“Count the Notes: Histogram‑Based Supervision for Automatic Music Transcription.”

### Transcription Examples
See the [project page](https://yoni-yaffe.github.io/count-the-notes/) for video demos and transcription results produced by CountEM.

## Features

- Train AMT models using weak, alignment-free supervision
- Leverages only histogram-level supervision (note counts)
- Built on top of **Onsets & Frames** and **Unaligned Supervision** codebases
- Comes with pre-trained models and scripts for training, inference, and data conversion

### Implementation
The implementation is based on the following projects:

* **[Unaligned Supervision](https://github.com/benadar293/benadar293.github.io)** – ICML 2022, alignment‑free training utilities.
* **[Onsets & Frames](https://github.com/jongwook/onsets-and-frames)** – the canonical PyTorch baseline for Onsets and Frames architecture.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Quick Start](#quick-start)
3. [Training & Inference](#training--inference)
4. [Bootstrap Checkpoint](#bootstrap-checkpoint)
5. [Credits & Citation](#credits--citation)
6. [License](#license)

---

## Repository Structure

```
.
├── train.py                           # ▶ Entry point: training
├── inference.py                       # ▶ Entry point: inference/evaluation
├── ckpts/                             # Contains a *link* to download a synthetic‑data transcriber
│   └── checkpoint_link.txt            #  (see Bootstrap Checkpoint)
├── datasets/                          # Directory that should contain the datasets. You can see the expected format in datasets/README.md
├── onsets_and_frames/                 # Upstream architecture (lightly modified)
├── conversion_maps/                   # Instrument ↔︎ MIDI helpers
├── scripts/                           # Utility scripts
│   ├── make_pitch_shifted_copies.py   # Data augmentation script
│   └── make_parsed_tsv_from_midi.py   # MIDI → TSV label conversion
├── NoteEM_tsv/                        # directory for midi labels in tsv format (used for training and eval)
├── static/ , index.html               # Demo / docs site assets (optional)
├── requirements.txt                   # Python dependencies
├── LICENSE.md
└── README.md
```

---

## Quick Start

> Requires **Python ≥3.8** and a GPU with CUDA 11+.

We provide both pyproject.toml and requirements.txt

You can use uv for installation:

```bash
# 1) Create virtual environment and install dependancies using UV
uv sync
```

Or you install using pip.

```bash
# 1) Create & activate a virtual‑env (recommended)
python -m venv .venv
source .venv/bin/activate        # (or venv\Scripts\activate on Windows)

# 2) Install dependencies
pip install -r requirements.txt

```

---
## Bootstrap Checkpoint

We do **not** ship large model weights.
Instead, the folder `ckpts/` contains a small text file `checkpoint_link.txt` with a download link to a transcriber trained purely on synthetic MIDI renderings.


---

## Training & Inference
### Training

```bash
# Show full help
python train.py -h

# Example usage
python train.py \
  --logdir "$LOGDIR" \
  --dataset-name <your_dataset_name> \
  --batch-size 8 \
  --transcriber-ckpt ckpts/model-70.pt
```

- **Audio data**: by default the script expects your audio under  
  `datasets/<your_dataset_name>/noteEM_audio`.  
  Override with `--data-dir-path` if you’ve moved your datasets elsewhere.  
  See [datasets/README.md](datasets/README.md) for the exact audio-folder layout.

- **MIDI/TSV labels**: by default it reads label files from  
  `NoteEM_tsv/<your_dataset_name>`.  
  You can override with `--tsv-dir`.  
  These must be in TSV format (`onset,offset,note,velocity,instrument`);  
  use the conversion tool at  
  `scripts/make_parsed_tsv_from_midi.py` to generate them from your `.mid` files.  
  See [NoteEM_tsv/README.md](NoteEM_tsv/README.md) for details on the TSV schema.

### Inference

```bash
# List available flags
python inference.py -h
```
---

## Evaluation Script (`eval.py`)

We provide a standalone evaluation utility to assess the accuracy of transcribed MIDI files against reference MIDI or TSV annotations using both **note-level** and **frame-level** metrics.

This is useful for benchmarking transcribers outside the training pipeline — for example, comparing inference results from `inference.py` with ground truth.

### Usage

```bash
python eval.py \
  --transcribed-dir evaluation/midis_no_pitch_shift_transcribed \
  --reference-dir evaluation/new_inference_dir/results \
  --outfile results.txt \
  --tolerance 0.05 \
  --shift 0.0 \
  --parallel \
  --max-workers 8
```

### Arguments

- `--transcribed-dir`:  
  Path to a folder containing **transcribed** `.mid` or `.midi` files, typically produced by a model (e.g., via `inference.py`).  
  Filenames must match or sort consistently with the reference files.

- `--reference-dir`:  
  Path to a folder with **reference MIDI or TSV files**.  
  Supported formats:
  - `.mid` or `.midi` — standard MIDI files
  - `.tsv` — 5-column text files (`onset`, `offset`, `pitch`, `velocity`, `instrument`)  
    Must be tab-delimited and compatible with `mir_eval`.

- `--outfile`:  
  Optional path to a `.txt` file where summary metrics will be saved.  
  A CSV file with per-piece scores will also be generated alongside it.

- `--tolerance`:  
  Onset tolerance in seconds for matching (default: `0.05`).

- `--shift`:  
  Uniform time shift (in seconds) to apply to the reference labels.  
  Useful when aligning outputs from different renderings.

- `--parallel`:  
  Use multiprocessing for faster evaluation over many files.

- `--max-workers`:  
  Number of CPU workers for parallel mode (default: `4`).

### Input Format

- The evaluation assumes that the transcribed and reference files match **by sorted filename order**.
- If using `.tsv` files in the reference directory, you can generate them using:

```bash
python scripts/make_parsed_tsv_from_midi.py --input-dir your_midis --output-dir NoteEM_tsv/your_dataset
```

<!-- ## Credits & Citation

If you build on this work, please cite our paper **and** the upstream repos we extend.

```bibtex
@inproceedings{yaffe2025countem,
  title     = {Count the Notes: Histogram‑Based Supervision for Automatic Music Transcription},
  author    = {Jonathan Yaffe and Ben Maman and Meinard Müller and Amit Bermano},
  booktitle = {Proc. ISMIR},
  year      = {2025}
}
``` -->

<!-- * Unaligned Supervision for AMT in the Wild (ICML 2022) — Maman & Bermano.
* Onsets & Frames (ISMIR 2018) — Hawthorne *et al.* -->

---

## License

© Jonathan Yaffe (Tel Aviv University), Ben Maman (International Audio Laboratories Erlangen, Germany),
Meinard Müller (International Audio Laboratories Erlangen, Germany), Amit Bermano (Tel Aviv University) 2025.

This project is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to **share** and **adapt** the material for any purpose, provided that:

1. **Attribution** — You give appropriate credit to the authors.
2. **ShareAlike** — If you remix, transform, or build upon the work, you must distribute your contributions under the same license.

See the full text in [LICENSE.md](LICENSE.md).

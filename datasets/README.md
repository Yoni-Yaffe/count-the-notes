# Datasets Directory

This directory contains datasets used for training and evaluation. Each dataset should follow the structure below, organized by pitch-shifted versions.

## Directory Structure

```
datasets/
└── noteEM_audio/
    ├── DATASET_NAME#0     # ▶ Original dataset (no pitch shift)
    ├── DATASET_NAME#-1    # ▶ Pitch shifted down by 1 semitone
    ├── DATASET_NAME#1     # ▶ Pitch shifted up by 1 semitone
    ├── DATASET_NAME#-2    # ▶ Pitch shifted down by 2 semitones
    └── ...                # ▶ Additional pitch-shifted versions
```

## Format Notes

- Each subdirectory inside `noteEM_audio/` represents a variant of the dataset with a specific pitch shift.
- The suffix `#N` indicates a pitch shift of **N semitones**:
  - `#0`: No shift (original)
  - Positive N: Shifted **up**
  - Negative N: Shifted **down**
- If pitch shifting is not applied, you only need to include `DATASET_NAME#0`.

# NoteEM_tsv Directory

This directory contains MIDI label files converted into a TSV (Tab-Separated Values) format, used for training and evaluation. The TSVs are generated from MIDI files using `scripts/make_parsed_tsv_from_midi.py`.

Each line in a TSV file represents a single note event with the following fields:

```
onset    offset    note    velocity    instrument
```

## Directory Structure

```
NoteEM_tsv/
└── DATASET_NAME/
    ├── file1.tsv
    ├── file2.tsv
    └── ...
```

- Each `.tsv` file corresponds to a `.mid`/`.midi` file in your source MIDI dataset.
- The filename structure and hierarchy mirrors that of the MIDI source folders used during conversion.

## TSV Format

Each row in a TSV file contains:

| Column      | Description                       |
|-------------|-----------------------------------|
| `onset`     | Note-on time in seconds           |
| `offset`    | Note-off time in seconds          |
| `note`      | MIDI pitch number (0–127)         |
| `velocity`  | Note velocity (0–127)             |
| `instrument`| MIDI program number (0–127)       |

> All values are floating-point except `note`, `velocity`, and `instrument`, which are integers.

## Generating TSVs from MIDI

Use the conversion script provided:

```bash
python scripts/make_parsed_tsv_from_midi.py
```

You can specify the MIDI source directory by editing the `midi_src_pth` variable in the script:

```python
midi_src_pth = '/path/to/midi/files'
target = 'NoteEM_tsv'
create_tsv_for_single_group(midi_src_pth, target)
```

For batch conversion of multiple datasets:

```python
midi_src_path_list = ['/path/to/set1', '/path/to/set2']
create_tsv_for_multiple_groups(midi_src_path_list, target)
```

## Notes

- Drum tracks (MIDI channel 9) are skipped during parsing.
- Sustain pedal effects are handled: notes are extended according to pedal state.
- You can override the instrument using `force_instrument` if needed.
- Files with pitch shifts should be matched manually to their audio counterparts (e.g., if you're training with pitch-shifted data).

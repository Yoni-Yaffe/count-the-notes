'''
Convert midi to a note-list representation in a tsv file. Each line in the tsv file will contain
information of a single note: onset time, offset time, note number, velocity, and instrument.
'''


from tqdm import tqdm
import numpy as np
import os
import warnings
from onsets_and_frames.midi_utils import parse_midi_multi
warnings.filterwarnings("ignore")

def midi2tsv_process(midi_path, target_path, shift=0, force_instrument=None):
    midi = parse_midi_multi(midi_path, force_instrument=force_instrument)
    print(target_path)
    if shift != 0:
        midi[:, 2] += shift
    np.savetxt(target_path, midi,
               fmt='%1.6f', delimiter='\t', header='onset,offset,note,velocity,instrument')



def create_tsv_for_single_group(midi_src_pth, target):
    FORCE_INSTRUMENT = None
    # piece = midi_src_pth.split(os.sep)[-1]
    piece = os.path.basename(midi_src_pth)
    os.makedirs(target + os.sep + piece, exist_ok=True)
    for f in tqdm(os.listdir(midi_src_pth)):
        if f.endswith('.mid') or f.endswith('.MID') or f.endswith('.midi'):
            # print(f)
            try:
                midi2tsv_process(midi_src_pth + os.sep + f,
                                target + os.sep + piece + os.sep + f.replace('.midi', '.tsv').replace('.mid', '.tsv').replace('.MID', '.tsv'),
                                force_instrument=FORCE_INSTRUMENT)
            except Exception as e:
                print(f"Error in {f}: {e}")

def create_tsv_for_multiple_groups(midi_src_pth_list, target):
    for midi_src in midi_src_pth_list:
        print(f"Creating tsv for group {midi_src}")
        create_tsv_for_single_group(midi_src, target)


if __name__ == "__main__":

    midi_src_pth = '/midi_src_path'

    target = 'NoteEM_tsv'
    create_tsv_for_single_group(midi_src_pth, target)
    # create_tsv_for_multiple_groups(midi_src_path_list, target)


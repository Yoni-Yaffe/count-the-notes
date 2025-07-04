'''
Create pitch shifted copies of the performance, from -5 to 5 semitones (11 copies).
'''


import os
from glob import glob

# src_dir = '/path/to/performance'
src_dir = 'new_samples_2201'
target_root = 'NoteEM_audio'
file_type = '.flac'


audio_src_files = glob(src_dir + '/**/*' + file_type, recursive=True)
audio_src_files = sorted(audio_src_files)

print('Beginning pitch shift from', src_dir)
for f in audio_src_files:
    print(f)
    f_split = f.split(os.sep)
    piece, part = f_split[-2:]
    try:
        assert os.sep.join(f_split[: -1]) == src_dir
    except AssertionError as e:
        print(os.sep.join(f_split[: -1]))
        print(src_dir)
        raise e
    for shift in range(-5, 6):
        print(shift)
        os.makedirs(target_root + os.sep + piece + '#' + str(shift), exist_ok=True)
        suffix = part[-len(file_type):]
        assert suffix == file_type
        f_target1 = target_root + os.sep + piece + '#' + str(shift) + os.sep + part.replace(file_type, '#{}.flac'.format(shift))
        f_target2 = target_root + os.sep + piece + '#' + str(shift) + os.sep + part[: -len(file_type)] + '#{}.flac'.format(shift)
        assert f_target1 == f_target2
        f_target = f_target1
        command = 'sox \"' + f + '\" -r 16000 \"' + f_target + '\" pitch {}'.format(100 * shift)

        # if you want to add a small shift (<= 0.1 semitone) use this command instead:
        # small_shift = np.random.randint(-10, 11)
        # command = 'sox \"' + f + '\" -r 16000 \"' + f_target + '\" pitch {}'.format(100 * shift + small_shift)

        print('command:', command)
        os.system(command)

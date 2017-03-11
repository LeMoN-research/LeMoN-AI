import c3d
import numpy as np
import subprocess
import os
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile

def positions_to_moves(positions):
    moves = np.array([np.subtract(positions[i+1, :, :3], positions[i, :, :3]) for i in range(len(positions)-1)])
    return moves

def get_sound_from_vid(vid_path, save_dir="audio/"):
    name = vid_path.split("/")[-1].split(".")[0]
    command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(vid_path, save_dir+name+".wav")
    subprocess.call(command, shell=True)
    return save_dir+name+".wav"

def get_sound_from_wav(wav_path, split_second=60, sr=44100):
    rec, sr = librosa.load(wav_path, sr=sr)
    pows_in_split = int(sr/split_second)
    pieces = np.array([rec[i:i+pows_in_split] for i in range(0, len(rec)-pows_in_split, pows_in_split)])
    return pieces

def all_videos_to_sound(vid_dir="videos/"):
    for file in os.listdir(vid_dir):
        get_sound_from_vid(vid_dir+file, "audio/")

### Dont use yet
def get_data(c3d_path="c3d/", audio_path="audio/"):
    c3ds = []; audio = []; names = []
    for file in os.listdir(c3d_path):
        with open(c3d_path+file, "rb") as handle:
            reader = c3d.Reader(handle)
            c3ds.append(np.array([x[1] for x in reader.read_frames()]))
        if file.replace("-C3D.c3d", "") in [x.replace("-HD.wav", "") for x in os.listdir(audio_path)]:
            audio.append(get_sound_from_wav(audio_path+file.replace("-C3D.c3d", "")+"-HD.wav"))
        else:
            audio.append(np.array([]))
        names.append(file.replace("-C3D.c3d", ""))
    return np.array(c3ds), np.array(audio), np.array(names)

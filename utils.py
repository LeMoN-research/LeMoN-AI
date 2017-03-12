import c3d
import numpy as np
import subprocess
import os
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from librosa import load, logamplitude
from librosa.feature import melspectrogram
from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed


def parallelization(func,massive,tq = True):
    
    num_cores = multiprocessing.cpu_count() # Число наших ядер
    results = np.array(Parallel(n_jobs=num_cores)(delayed(func)(i) for i in massive))
    return results


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
def get_data(npy_path):
    data_matr = np.load(npy_path)
    return np.array([np.array([y[0] for y in x]) for x in data_matr]),\
        np.array([np.array([y[1] for y in x]) for x in data_matr])

    
def get_spectrogram(y, sr=43680):
    S = melspectrogram(y, sr=sr, n_mels=200)
    log_S = logamplitude(S, ref_power=np.max)
    return log_S


def get_spectrs(music, block_size=20, dump_path="spectr/"):
    for ind, mus in enumerate(music):
        for i in range(0, len(music)-block_size, 1):
            spec = get_spectrogram(mus[i:i+block_size].reshape(-1), 43680).T 
            spec.dump("{}{}.{}.spec".format(dump_path, ind, i))
            
            
def iterate_minibatches(points, music, batch_size=100, block_size=20):
    p = Pool(32)
    blocks_left = [int(p.shape[0]/block_size) for p in points]
    blocks_total = blocks_left
    moves = np.array([positions_to_moves(x) for x in points])
    batch_count = 0
    train_moves = []; pred_move = []; start_pos = []; out_spec = []  
    while np.any(blocks_left != 0):
        inst = np.random.choice(range(len(blocks_left)))
        start = -blocks_left[inst]*block_size
        fin = start+block_size            
        
        if fin < moves.shape[0]:
            train_moves.append(moves[inst][start:fin-1])

            pred_move.append(moves[inst][fin])

            start_pos.append(points[inst][start])

            our_spec = np.load("spectr/{}.{}.spec".format(inst, blocks_total[inst]-blocks_left[inst]))
            out_spec.append(our_spec)
  

        blocks_left[inst] -= 1


        if len(train_moves) == batch_size:
            yield np.array(out_spec), np.array(train_moves), np.array(start_pos), np.array(pred_move)
            train_moves = []; pred_move = []; start_pos = []; out_spec = []  

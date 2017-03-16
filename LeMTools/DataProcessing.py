import numpy as np
import c3d
import subprocess
import os
import librosa

class C3DProcessor(object):
    """
    Description:
        :read/write c3d data
        :make this data as beautiful as Cara Delevingne (More detailed inforamtion bellow)
    """

    def __init__(self, points_amount=38, frames=None):
        self.points_amount = points_amount
        self.frames = frames
    
    ### R/W operations 
    @staticmethod
    def read(path_to_read):
        reader = c3d.Reader(open(path_to_read, 'rb'))
        coordinates = np.array(map(lambda x: x[1], list(reader.read_frames())))
        return coordinates[:, :, :3]

    def write(self, path_to_write, matrix, additional_matrix=None):
        assert np.ndim(matrix) == 3
        assert matrix.shape[2] == 3 and matrix.shape[1] == self.points_amount
        
        if additional_matrix == None:
            # define last two columns in the matrix of coordinates with zeros 
            additional_matrix = np.zeros(dtype="float32", shape=(matrix.shape[0], self.points_amount, 2))
            
        writer = c3d.Writer()
        # frames - generator of tuples (coord_matrix,  analog)
        frames = ((np.concatenate((matrix[i]*-1, additional_matrix[i]), axis=1), np.array([[]])) for i in range(len(matrix)))
        writer.add_frames(frames)
        
        with open(path_to_write, 'wb') as h:
            writer.write(h)
            
    """
    Beautification:
        Now a little more details. Most of the C3D files have a problem,
        where some of their points coordinates go to zero and then back to the
        normal coordinates. It's caused by the fact, that that point was not seen
        by at least to cameras at that moment. 
        This is here to fix it.
    """
    
    def set_frames(self,frames):
        self.frames = frames
    
    def main(self):
        assert self.frames != None
        return self.smooth(self.add_extra_points(self.frames))

    # smooth points trajectory - for cases when point completely disappears
    # fill in with data from previous frame
    # TODO rewrite smooth function (include extrapolating)
    @staticmethod
    def smooth(frames, verbose=False):
        zero_points = 0
        
        for frame_index, frame in enumerate(frames):
            for point_index, point in enumerate(frame):
                if point[0] == point[1] == point[2] == 0:
                    frames[frame_index][point_index] = frames[frame_index-1][point_index]
                    zero_points += 1

        if verbose:
            print zero_points

        return frames

    @staticmethod
    def add_extra_points(frames):
        extra_points = [group_center(make_group([11, 18], frame)) for frame in frames]
        new_frames = np.array([np.concatenate((frames[i], np.array([extra_points[i]]))) for i in range(len(frames))])
        return new_frames

    @staticmethod
    def extrapolate(frame, (i, j), deg=10):
        assert frame.shape[1] == 3 and np.ndim(frame) == 2
        # get polynomial coefficients
        poly_coef = lambda a, b: np.polyfit(a, b, deg)
        # get polynomial itself
        polynomial = lambda coefs: np.poly1d(coefs)

        get_values = lambda func: [func(k) for k in range(i, j)]

        length = frame.shape[0] + j - i
        indexes = np.concatenate((np.array(range(0, i)), np.array(range(j, length))))

        funcs = []

        for m in range(3):
            column = frame[:, m]
            p = polynomial(poly_coef(indexes,column))
            funcs.append(p)

        return np.array([get_values(funcs[0]), get_values(funcs[1]), get_values(funcs[2])])  

    @staticmethod
    def moves_to_position(start_pos, matrix_of_movements):
        assert np.ndim(matrix_of_movements) == 3
        assert matrix_of_movements.shape[2] == 3

        return start_pos + np.sum(matrix_of_movements, axis=0)
    
    @staticmethod
    def positions_to_moves(positions):
        moves = np.array([np.subtract(positions[i+1, :, :3], positions[i, :, :3]) for i in range(len(positions)-1)])
        return moves


class MediaProcessor(object):
    """
    Description:
    * Get sound from training videos
    * Sound processing
    """
    @staticmethod
    def get_wav_from_vid(vid_path, save_dir="audio/"):
        '''
        vid_path = path to video
        save_dir = directory to save results to
        '''
        name = vid_path.split("/")[-1].split(".")[0]
        command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(vid_path, save_dir+name+".wav")
        subprocess.call(command, shell=True)
        return save_dir+name+".wav"

    @staticmethod
    def get_sound_from_wav(wav_path, split_second=60, sr=44100):
        '''
        wav_path = path to wav (duh...)
        split_second = amount of sound blocks in one second
        sr = sample rate of recording
        '''
        rec, sr = librosa.load(wav_path, sr=sr)
        pows_in_split = int(sr/split_second)
        pieces = np.array([rec[i:i+pows_in_split] for i in range(0, len(rec)-pows_in_split, pows_in_split)])
        return pieces

    @staticmethod
    def all_videos_to_wav(vid_dir="videos/", save_dir="audio"):
        '''
        Translate all videos in directory to wav
        '''
        for file in os.listdir(vid_dir):
            get_sound_from_vid(vid_dir+file, "audio/")

    @staticmethod
    def get_spectrogram(y, sr=43680):
        S = melspectrogram(y, sr=sr, n_mels=200)
        log_S = logamplitude(S, ref_power=np.max)
        return log_S

    @staticmethod
    def get_spectrs(music, block_size=20, sr=43680):
        '''
        Get spectrograms from splitted sound
        music = sound, created by get_sound_from_wav
        block_size = how many splits to convert into spectrogram
        sr = sample rate
        '''
        spectrs = []
        for ind, mus in enumerate(music):
            for i in range(0, len(music)-block_size, 1):
                spectrs.append(self.get_spectrogram(mus[i:i+block_size].reshape(-1), sr).T) 
        return spectrs

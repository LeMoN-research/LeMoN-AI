import numpy as np
import c3d
import subprocess
import os
from librosa import load, logamplitude
from librosa.feature import melspectrogram


def moves_to_position(start_pos, matrix_of_movements):
    assert np.ndim(matrix_of_movements) == 3
    assert matrix_of_movements.shape[2] == 3
    return start_pos + np.sum(matrix_of_movements, axis=0)


def positions_to_moves(positions):
    moves = np.array([np.subtract(positions[i + 1, :, :3], positions[i, :, :3]) for i in range(len(positions) - 1)])
    return moves


class C3DProcessor(object):
    """
    Description:
    :read/write c3d data
    :make this data as beautiful as Cara Delevingne (More detailed information bellow)
    """

    def __init__(self, points_amount=38, frames=None):
        self.points_amount = points_amount
        self.frames = frames

    # R/W operations
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
        frames = ((np.concatenate((matrix[i] * -1, additional_matrix[i]), axis=1), np.array([[]]))
                  for i in range(len(matrix)))
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

    def clean(self, frames):
        return self.smooth(self.add_extra_points(self.frames))

    # smooth points trajectory - for cases when point completely disappears
    # fill in with data from previous frame

    @staticmethod
    def find_zero_sections(frames):
        """
        Find zero sections in points trajectories.
        :param frames: original c3d frames
        :return: coordinates of zero sections for every point
        """
        frames = np.transpose(frames, [1, 0, 2])

        all_points = []
        for points in frames:
            zeros = np.where(points == 0)[0]
            zeros = np.array([zeros[i] for i in range(0, len(zeros), 3)])
            sections = []
            if zeros.shape[0] != 0:
                start = zeros[0]
                for i in range(len(zeros) - 1):
                    if zeros[i + 1] - zeros[i] != 1:
                        sections.append((start, zeros[i] + 1))
                        start = zeros[i + 1]
                sections.append((start, zeros[-1] + 1))
            all_points.append(sections)
        return np.array(all_points)

    def smooth(self, frames, sections, n=20):
        """
        Fill all zero sections with extrapolated numbers.
        :param frames: original c3d frames
        :param sections: coordinates of zero sections for every point
        :param n: number of points left and right of the section to fit into extrapolate function
        :return: filled version
        """
        assert frames.shape[1] == sections.shape[0]

        not_neg = lambda x: 0 if x < 0 else x
        not_max = lambda x, y: y - 1 if x >= y else x

        frames = np.transpose(frames, [1, 0, 2])

        for frame_idx in range(len(frames)):
            for section in sections[frame_idx]:
                # values on the edges of zero section (to find coordinates in slice later)
                edge_dots = (frames[frame_idx][section[0] - 1],
                             frames[frame_idx][not_max(section[1], frames[frame_idx].shape[0])])
                # our slice that we would want to fill
                our_stuff = frames[frame_idx][not_neg(section[0] - n):not_max(section[1] + n,
                                                                              frames[frame_idx].shape[0])]
                # if zero section is not in the end
                if np.sum(edge_dots[1]) != 0:
                    edge_pos = (np.where(our_stuff == edge_dots[0])[0][0], np.where(our_stuff == edge_dots[1])[0][0])

                    before_section = np.array(list(filter(lambda x: np.sum(x) != 0, our_stuff[:edge_pos[0] + 1])))
                    after_section = np.array(list(filter(lambda x: np.sum(x) != 0, our_stuff[edge_pos[1]:])))

                    slice_ = np.concatenate((before_section, after_section))

                    zeros_start = edge_pos[0] + 1
                    zeros_end = edge_pos[0] + (section[1] - section[0]) + 1
                else:
                    slice_ = np.array(list(filter(lambda x: np.sum(x) != 0,
                                                  frames[frame_idx][not_neg(section[0] - n):])))
                    zeros_start = slice_.shape[0]
                    zeros_end = zeros_start + (section[1] - section[0])

                extrapolated = self.extrapolate(slice_, (zeros_start, zeros_end))

                frames[frame_idx][section[0]:section[1]] = extrapolated
        return frames.transpose([1, 0, 2])

    def add_extra_points(self, frames):
        extra_points = [self.group_center(self.make_group([11, 18], frame)) for frame in frames]
        new_frames = np.array([np.concatenate((frames[i], np.array([extra_points[i]]))) for i in range(len(frames))])
        return new_frames

    @staticmethod
    def extrapolate(frame, idx, deg=10):
        i = idx[0];
        j = idx[1]
        assert frame.shape[1] == 3 and np.ndim(frame) == 2
        # get polynomial coefficients
        poly_coef = lambda a, b: np.polyfit(a, b, deg)
        # get polynomial itself
        polynomial = lambda coefs: np.poly1d(coefs)
        # sort of word of wisdom
        get_values = lambda func: [func(k) for k in range(i, j)]

        length = frame.shape[0] + j - i
        indexes = np.concatenate((np.array(range(0, i)), np.array(range(j, length))))

        funcs = []
        for m in range(3):
            column = frame[:, m]
            p = polynomial(poly_coef(indexes, column))
            funcs.append(p)

        dots = np.array([get_values(funcs[0]), get_values(funcs[1]), get_values(funcs[2])])

        return np.array([[dots[0, x], dots[1, x], dots[2, x]] for x in range(len(dots[0]))])

class MediaProcessor(object):
    """
    Description:
    :Get sound from training videos
    :Sound processing
    """

    @staticmethod
    def get_wav_from_vid(vid_path, save_dir="audio/"):
        """
        vid_path = path to video
        save_dir = directory to save results to
        """
        name = vid_path.split("/")[-1].split(".")[0]
        command = "ffmpeg -i {0} -ab 160k -ac 2 -ar 44100 -vn {1}".format(vid_path, save_dir + name + ".wav")
        subprocess.call(command, shell=True)
        return save_dir + name + ".wav"

    @staticmethod
    def get_sound_from_wav(wav_path, split_second=60, sr=44100):
        """
        wav_path = path to wav (duh...)
        split_second = amount of sound blocks in one second
        sr = sample rate of recording
        """
        rec, sr = load(wav_path, sr=sr)
        pows_in_split = int(sr / split_second)
        pieces = np.array([rec[i:i + pows_in_split] for i in range(0, len(rec) - pows_in_split, pows_in_split)])
        return pieces

    def all_videos_to_wav(self, vid_dir="videos/", save_dir="audio/"):
        """
        Translate all videos in directory to wav
        """
        for file in os.listdir(vid_dir):
            self.get_sound_from_vid(vid_dir + file, save_dir)

    @staticmethod
    def get_spectrogram(y, sr=43680):
        S = melspectrogram(y, sr=sr, n_mels=200)
        log_S = logamplitude(S, ref_power=np.max)
        return log_S

    def get_spectrs(self, music, block_size=20, sr=43680):
        """
        Get spectrograms from splitted sound
        music = sound, created by get_sound_from_wav
        block_size = how many splits to convert into spectrogram
        sr = sample rate
        """
        spectrs = []
        for ind, mus in enumerate(music):
            for i in range(0, len(music) - block_size, 1):
                spectrs.append(self.get_spectrogram(mus[i:i + block_size].reshape(-1), sr).T)
        return spectrs

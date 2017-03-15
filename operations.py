import numpy as np
import c3d
from configuration import *
from tqdm import tqdm
import pickle as pkl


class DataProcessor(object):
    """
    description of the class:
    * read function reads file by given path and returns the matrix of movements
    * write function creates new c3d file by given matrix of movements
    * make_group function takes indexes of certain points and combines them into a group
    """

    def __init__(self, points_amount=38):
        self.points_amount = points_amount

    def read(self, path_to_read):
        reader = c3d.Reader(open(path_to_read, 'rb'))
        coords = np.array(map(lambda x: x[1], list(reader.read_frames())))
        return coords[:, :, :3]

    def write(self, path_to_write, matrix, additional_matrix=None):
        assert np.ndim(matrix)== 3
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


def make_group(labels, frame):
    # make_group function takes indexes of certain points and combines them into a group
    return frame[labels, :]


def group_center(group):
    return np.mean(group, axis=0)


class Atomy(object):
    """
    incapsulates methods for estimating penalty to a frame of points for not looking humanlike
    - generates a metrics from training data
    - - average distance of points in a group from the group center
    - - average distance between two points at the body joint
    """

    def __init__(self, frames, all_groups_labels=groups, all_joints=joints, build_means=True):
        
        # group labels - list of groups (another list of indexes of points in that group (i.e. left calf))
        self.all_groups_labels = all_groups_labels 
        # joint labels - list of pairs (indexes of two points that should form a joint)
        self.all_joints = all_joints
        
        if build_means:
            self.group_distances, self.joint_distances = self.compute_mean_matrices(frames)
            pkl.dump((self.group_distances, self.joint_distances), open('mean_matrices.pkl','wb'))
            
        else:
            self.group_distances, self.joint_distances = pkl.load(open('mean_matrices.pkl','rb'))

    def dist(self, (point_1, point_2)):
        assert len(point_1) == len(point_2) == 3, 'incorrect shapes '+ str(point_1.shape)+' and '+str(point_2.shape)
        return np.sqrt(np.sum((point_2-point_1)**2))

    def dists_from_center(self,center,group):
        return [self.dist((center, x)) for x in group]

    def update_centers(self, frame):
        # updates coordinates of group center, in order for them to correspond to currently evaluated frame
        return [group_center(make_group(labels, frame)) for labels in self.all_groups_labels]
    
    # initialization - iterates over all training data once to estimate appropriate (humanlike) restrictions
    # on relative movements of points
    def compute_mean_matrices(self, frames):
        # center - group points
        length = len(frames)
        group_distances = [np.zeros(shape=(len(n),)) for n in self.all_groups_labels]
        joint_distances = np.zeros(shape=(len(self.all_joints)),)
        
        for frame in tqdm(frames):
            group_dists, joint_dists = self.update_dists(frame) 
            
            joint_distances = [joint_distances[i] + joint_dists[i] for i in range(len(self.all_joints))]
            group_distances = [group_distances[i] + group_dists[i] for i in range(len(self.all_groups_labels))]
        
        return tuple(np.array(group_distances)/length, np.array(joint_distances)/length)
    
    # updates lists of distances of points
    # - from the centers of their respective groups
    # - of joints from each other
    def update_dists(self, frame):
        centers = self.update_centers(frame)
        group_dists = np.array([self.dists_from_center(centers[i], make_group(self.all_groups_labels[i], frame))
                          for i in range(len(centers))])
        
        joint_dists = np.array(map(self.dist, map(lambda x: make_group(x,frame), self.all_joints)))
        
        return tuple(group_dists, joint_dists)

    # calculates penalty based on deviation of point distances from mean point distances
    def compute_extra_error(self,frame):
        e = lambda l: [item for sublist in l for item in sublist]
        ms = lambda l: np.mean(np.square(l))
        group_dists, joint_dists = self.update_dists(frame)
        error = ms(e(group_dists-self.group_distances)) + np.mean(joint_dists-self.joint_distances)
        return error 

    def frames_error(self, frames):
        return np.mean(np.array([self.compute_extra_error(frame) for frame in frames]))


class Preprocess(object):
    """
    beautification of data:
    - create additional points
    - - one point for the neck
    - fill in data on points that are missing from some of the frames
    """

    def __init__(self, frames):
        assert np.ndim(frames) == 3 and frames.shape[2] == 3
        self.frames = frames

    def main(self):
        return self.smooth(self.add_extra_points(self.frames))

    # smooth points trajectory - for cases when point completely dissapears
    # fill in with data from previous frame
    def smooth(self, frames, verbose=False):
        zero_points = 0
        
        for frame_index, frame  in enumerate(frames):
            for point_index, point in enumerate(frame):
                if point[0] == point[1] == point[2] == 0:
                    frames[frame_index][point_index] = frames[frame_index-1][point_index]
                    zero_points += 1

        if verbose:
            print zero_points

        return frames

    def add_extra_points(self,frames):
        extra_points = [group_center(make_group([11,18],frame)) for frame in frames]
        new_frames = np.array([np.concatenate((frames[i],np.array([extra_points[i]]))) for i in range(len(frames))])
        return new_frames

    def extrapolate(self, frame, (i,j), deg=10):
        assert frame.shape[1] == 3 and np.ndim(frame) == 2
        # get polynomial coefficients
        poly_coef = lambda a, b: np.polyfit(a,b,deg)
        # get polynomial itself
        polynomial = lambda coefs: np.poly1d(coefs)

        get_values = lambda func: [func(k) for k in range(i,j)]

        length = frame.shape[0] + j - i
        indexes = np.concatenate((np.array(range(0, i)), np.array(range(j,length))))

        funcs = []

        for m in range(3):
            column = frame[:,m]
            p = polynomial(poly_coef(indexes,column))
            funcs.append(p)

        return np.array([get_values(funcs[0]), get_values(funcs[1]), get_values(funcs[2])])  

    def moves_to_position(self, start_pos, matrix_of_movements):
        assert np.ndim(matrix_of_movements) == 3
        assert matrix_of_movements.shape[2] == 3

        return start_pos + np.sum(matrix_of_movements, axis=0)

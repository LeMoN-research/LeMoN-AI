import numpy as np
import pickle as pkl
from tqdm import tqdm
from configuration import *


def make_group(labels, frame):
    # make_group function takes indexes of certain points and combines them into a group
    return frame[labels, :]


def group_center(group):
    return np.mean(group, axis=0)


class Skeleton(object):
    """
    encapsulates methods for estimating penalty to a frame of points for not looking humanlike
    :generates a metrics from training data
        :average distance of points in a group from the group center
        :average distance between two points at the body joint
    """

    def __init__(self, frames, all_groups_labels=groups, all_joints=joints, build_means=True):
        
        # group labels - list of groups (another list of indexes of points in that group (i.e. left calf))
        self.all_groups_labels = all_groups_labels 
        # joint labels - list of pairs (indexes of two points that should form a joint)
        self.all_joints = all_joints
        
        if build_means:
            self.group_distances, self.joint_distances = self.compute_mean_matrices(frames)
            pkl.dump((self.group_distances, self.joint_distances), open('mean_matrices.pkl', 'wb'))
            
        else:
            self.group_distances, self.joint_distances = pkl.load(open('mean_matrices.pkl', 'rb'))

    @staticmethod
    def dist((point_1, point_2)):
        assert len(point_1) == len(point_2) == 3, 'incorrect shapes ' + str(point_1.shape) + ' and '+str(point_2.shape)
        return np.sqrt(np.sum((point_2-point_1)**2))

    def dists_from_center(self, center, group):
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
        group_dists = np.array([self.dists_from_center(centers[i], make_group(self.all_groups_labels[i], frame))\
                                for i in range(len(centers))])
        
        joint_dists = np.array(map(self.dist, map(lambda x: make_group(x, frame), self.all_joints)))
        
        return tuple(group_dists, joint_dists)

    # calculates penalty based on deviation of point distances from mean point distances
    def compute_extra_error(self, frame):
        e = lambda l: [item for sublist in l for item in sublist]
        ms = lambda l: np.mean(np.square(l))
        group_dists, joint_dists = self.update_dists(frame)
        error = ms(e(group_dists-self.group_distances)) + np.mean(joint_dists-self.joint_distances)
        return error 

    def frames_error(self, frames):
        return np.mean(np.array([self.compute_extra_error(frame) for frame in frames]))

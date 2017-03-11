import numpy as np
import c3d
'''
description of the class below:
* read function reads file by given path and returns the matrix of movements
* write function creates new c3d file by given matrix of movements 
* make_group function takes indexes of certain points and combines them into a group 
'''
class DataProcessor(object):
    
    def __init__(self, points_amount=38):
        self.points_amount = points_amount
        
        
    def read(self, path_to_read):
        reader = c3d.Reader(open(path_to_read, 'rb'))
        coords = np.array(map(lambda x: x[1], list(reader.read_frames())))
        return coords[:,:,:3]
        
        
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
            
            
    def smooth(self,verbose=False):
        zero_points = 0
        frames = coords_x[:,:,:3]
        for frame_index, frame  in enumerate(frames):
            for point_index, point in enumerate(frame):
                if point[0] == point[1] == point[2] == 0:
                    frames[frame_index][point_index] = frames[frame_index-1][point_index]
                    zero_points += 1

        if verbose:
            print zero_points

        return frames
    
    
'''
description of the class below:
* make_group function takes indexes of certain points and combines them into a group 
'''
# TODO: 2 dimensional array problem (in __init__) 
class Atomy(object):  
    def __init__(self, frame, all_groups_labels):
        # iterating throw all groups labels and counting distances from each center to other points of the certain group
        centers = [self.group_center(self.make_group(labels, frame)) for labels in all_groups_labels]
        self.distances = np.array([self.dists_from_center(centers[i], self.make_group(all_groups_labels[i], frame))
                          for i in range(len(centers))])
    
    
    def dist(self, point_1, point_2):
        return np.sqrt(np.sum((point_2-point_1)**2))

    
    def dists_from_center(self,center,group):
        return [self.get_dist(center, x) for x in group]

    
    def make_group(self, labels, frame):
        return frame[labels, :]
        
                
    def group_center(self, group):
        return np.mean(group, axis=1)
    
    
    def error(self, dist, epsilon_min=50, epsilon_max=5):
        if dist > epsilon_max:
            return np.exp(dist - epsilon_max)
        
        elif dist < epsilon_min:
            return np.exp(epsilon_min*10 - dist)
        
        return 0

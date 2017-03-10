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
            
    
    def make_group(self, labels, matrix):
        return matrix[:, labels, :]

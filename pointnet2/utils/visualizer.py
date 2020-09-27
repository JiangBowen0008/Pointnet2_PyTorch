'''
A visualizer that displays the pointcloud with the predicted labels.
Notice:
    Use open3d 0.5.0 instead of 0.10.0 as the latest version has conflict
    with PyTorch
'''

import numpy as np
import open3d as o3d
import torch
import os
import errno
from colorsys import hsv_to_rgb


class Visualizer():
    def __init__(self, outputDir: str):
        '''
        Designates output directory
        '''
        self.outputDir = outputDir

    def show(self, pc: np.ndarray, labels: np.ndarray, save = False, saveName = None) -> str:
        '''
        Display an numpy array of points with labels as an o3d point cloud.

        Args:
            pc (np.ndarray) : Point cloud
            labels (np.ndarray) : Predicted labels
            save (bool) : Whether to save the point cloud
            saveName (str) : Name of the output file. Only needed if save is True.
        
        Return:
            (when output is True) str : Output filename (*.ply)
        '''
        # first converts to o3d type
        pc = self._toNumpy(pc)
        labels = self._toNumpy(labels)
        pointCloud = self._toPointCloud(pc = pc, labels = labels)

        # display
        o3d.visualization.draw_geometries([pointCloud])

        # save the point cloud if required
        if (save):
            return self._save(pointCloud, saveName)
        
        return ""

    def save(self, pc: np.ndarray, labels: np.ndarray, name: str) -> str:
        '''
        Helper function of _save method.
        '''

        # converts to the o3d type first
        pc = self._toNumpy(pc)
        labels = self._toNumpy(labels)
        pointCloud = self._toPointCloud(pc = pc, labels = labels)
        
        return self._save(pointCloud, name)
    
    def _save(self, pointCloud, name: str) -> str:
        '''
        Saves the numpy array into a point cloud file on local storage
        Args:
            pc (np.ndarray) : Point cloud
            labels (np.ndarray) : Predicted labels
        
        Return:
            str : Output filename (*.ply)
        '''

        # create the folder if it doesn't exist
        try:
            os.makedirs(self.outputDir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        filename = os.path.join(self.outputDir, name)
        o3d.write_point_cloud(filename + '.ply', pointCloud, write_ascii = True)
        o3d.write_point_cloud(filename + '.pcd', pointCloud, write_ascii = True)
        print("Files saved to %s." % filename)

        return filename

    def _toPointCloud(self, pc: np.ndarray, labels: np.ndarray) -> o3d.geometry.PointCloud:
        '''
        Converts an numpy array of points with labels to an o3d point cloud.

        The point cloud is color coded using the prediction labels.

        Args:
            pc (np.ndarray) : Point cloud array
            labels (np.ndarray) : Predicted labels
        
        Return:
            o3d.geometry.PointCloud : o3d point cloud
        '''
        # read numpy array of the point cloud into o3d
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.Vector3dVector(pc[:, 0:3])

        # get colors from labels
        colors = self._colorize(labels)
        pc_o3d.colors = o3d.Vector3dVector(colors[:, 0:3])

        return pc_o3d        

    def _colorize(self, labels: np.ndarray) -> np.ndarray:
        '''
        Color label the labels given.

        It normalizes the label value and use the normalized values as the hue
        of the colors.

        Args:
            labels (np.ndarray) : labels of the point cloud of size [numPt]
        
        Return:
            np.ndarray : color labels (NumPy array of size [numPt, 3])
        '''
        numPts = labels.shape[0]

        colors = np.zeros((numPts, 3))              # three channel colors

        maxLabel = np.max(labels)
        minLabel = np.min(labels)
        rangeLabel = maxLabel - minLabel

        color_h = (labels - minLabel) / rangeLabel  # Normalizing hue within [0, 1]
        colors[..., 0] = color_h
        colors[..., 1] = 0.6                        # s
        colors[..., 2] = 0.6                        # v

        # converting from hsv to rgb
        hsv_to_rgb_ = lambda row : hsv_to_rgb(row[0], row[1], row[2])
        colors = np.apply_along_axis(hsv_to_rgb_, 1, colors)        
        print(colors.shape)
        return colors

    def _toNumpy(self, T):
        '''
        Utility conversion for common input types
        '''
        if isinstance(T, np.ndarray):
            _T = T
        elif isinstance(T, torch.Tensor):
            _T = T.cpu().numpy()
        else:
            print("Unsupported type: " + type(T))
            exit(-1)
        
        return _T    
    
    
'''
Test module
'''
if __name__=='__main__':
    vis = Visualizer("../outputs/temp")
    n = 1000
    pc = np.random.random((n,3))
    labels = np.random.randint(low = 0, high = 13, size = (n,))
    vis.save(pc, labels, "test")
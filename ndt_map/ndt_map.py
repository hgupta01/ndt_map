import re
import copy
import random
import time
import math
import numpy as np


class NDTCell():
    """
    NDT cell type object.
    """
    def __init__(self, nmin=5):
        self.point_mean_= np.zeros(3)
        self.point_cov_ = np.eye(3)
        self.minN_ = nmin # min number of points required
        self.has_gaussian_ = False
        self.points = []
    
    def addPoint(self, pnt):
        """
        Adding point to the list. input should be a Numpy vector
        """
        self.points.append(pnt)
    

    def calculateGaussian(self):
        """
        Calculate mean covarience and inverse covarience for the cell
        """        
        if (len(self.points) < self.minN_):
            return False
        
        self.point_mean_ = np.mean(self.points, axis=0)
        diff = self.points - self.point_mean_
        self.point_cov_ = (diff.T@diff)/(len(diff)-1)
        if (np.linalg.det(self.point_cov_) < 0 or np.linalg.det(self.point_cov_)<1e-20):
            return False

        try:
            icov = np.linalg.inv(self.point_cov_)
        except np.linalg.LinAlgError:
            return False

        self.has_gaussian_ = True
        
    def hasGaussian(self):
        return self.has_gaussian_
        
    def getCovarience(self):
        return self.point_cov_
    
    def getMean(self):
        return self.point_mean_


class HashGrid(object):
    """
    HashGrid3d is a a spatial index which can be used for creating 3D NDT maps.
    """
    def __init__(self, cell_size=0.5, nmin=5):
        self.cell_size = cell_size
        self.nmin=nmin
        self.grid = {}


    def key(self, point):
        """
        Key for dictionary. returns a tuple.
        """
        pnt = np.floor(point/self.cell_size)
        return (int(pnt[0]), int(pnt[1]), int(pnt[2]))
    
    def getCell(self, pnt):
        """
        Returns the cell at input point, if cell  
        """
        key_vector = self.key(pnt)
        if (key_vector not in self.grid):
            self.grid[key_vector] = NDTCell(self.nmin)
        return self.grid.get(key_vector)
    
    def hasCell(self, pnt):
        """
        Check if a cell is present at the enquiry point
        """
        key_vector = self.key(pnt)
        if (key_vector not in self.grid):
            return False
        return True
    
    def getGrid(self):
        return self.grid


class NDTMap():
    def __init__(self, cell_size=0.5, nmin=5):
        self.cell_size = cell_size
        self.map_ = HashGrid(self.cell_size, nmin)
        self.num_of_gaussians = 0
    
    def addPoint(self, pnt):
        """
        Adding point to the the map. Input is numpy vector 6
        """
        self.map_.getCell(pnt).addPoint(pnt)
        
    def addPointCloud(self, pnts):
        """
        Adding point cloud to the the map. Input is list of numpy vector n*6
        """
        for pnt in pnts:
            self.addPoint(pnt)
    
    def calculateGaussian(self):
        """
        Compute gaussian for each cell in map.
        """
        grid = self.map_.getGrid()
        for key in grid:
            grid[key].calculateGaussian()
            if grid[key].hasGaussian():
                self.num_of_gaussians = self.num_of_gaussians +1
        
    def getNumberOfGaussians(self):
        """
        Returns the total number of valid Gaussian
        """
        return self.num_of_gaussians
    
    def getMeanAndCov(self):
        means = []
        covs = []
        grid = self.map_.getGrid()
        for key in grid:
            if (grid[key].hasGaussian()):
                means.append(grid[key].getMean())
                covs.append(grid[key].getCovarience())

        return np.array(means), np.array(covs)
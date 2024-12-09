#!/usr/bin/env python
# -*- coding: utf-8 -*-


import psutil
import numpy as np
from sklearn.preprocessing import normalize


class PS(object):


    def __init__(self):
        self.M = None   # measurement matrix in numpy array
        self.L = None   # light matrix in numpy array
        self.N = None   # surface normal matrix in numpy array
        self.height = None  # image height
        self.width = None   # image width
        self.foreground_ind = None    # mask (indices of active pixel locations (rows of M))
        self.background_ind = None    # mask (indices of inactive pixel locations (rows of M))

    def load_lighttxt(self, filename=None):
        """
        Load light file specified by filename.
        The format of lights.txt should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.txt
        """
        self.L = psutil.load_lighttxt(filename)

    def load_lightnpy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.L = psutil.load_lightnpy(filename)

    def load_images(self, foldername=None, ext=None):
        """
        Load images in the folder specified by the "foldername" that have extension "ext"
        :param foldername: foldername
        :param ext: file extension
        """
        self.M, self.height, self.width = psutil.load_images(foldername, ext)

    def load_npyimages(self, foldername=None):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        self.M, self.height, self.width = psutil.load_npyimages(foldername)

    def load_mask(self, filename=None):
        """
        Load mask image and set the mask indices
        In the mask image, pixels with zero intensity will be ignored.
        :param filename: filename of the mask image
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        mask = psutil.load_image(filename=filename)
        mask = mask.reshape((-1, 1))
        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        psutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        psutil.save_normalmap_as_npy(filename=filename, normal=self.N, height=self.height, width=self.width)


    def solve(self):
        if self.M is None:
            raise ValueError("Measurement M is None")
        if self.L is None:
            raise ValueError("Light L is None")

        print("Before np.linalg.lstsq. L shape:", self.L.shape, "M shape:", self.M.shape)

        #############################################

        # Please write your code here to solve the surface normal N whose size is (p, 3) as disc

        # 转置self.L和self.M
        # self.L和self.M的数据格式并不符合np.linalg.lstsq的要求
        L_transposed = self.L.T
        M_transposed = self.M.T

        # Step 1: solve Ax = b
        # Hint: You can use np.linalg.lstsq(A, b) to solve Ax = b
        # 使用 np.linalg.lstsq 求解方程 Ax = b，这里 A 是转置后的L，b是转置后的M
        solution = np.linalg.lstsq(L_transposed, M_transposed)
        if solution[0] is None:
            raise ValueError("np.linalg.lstsq returned None for the solution")
        # self.N = ???
        # 得到表面法线 N，形状为 (p, 3)，其中 p 是像素数量
        self.N = solution[0].T

        print("After np.linalg.lstsq. N shape:", self.N.shape)

        # 检查 self.N 的形状是否符合预期
        if self.N.shape[0] != self.M.shape[0] or self.N.shape[1] != 3:
            raise ValueError(f"Unexpected shape of self.N: {self.N.shape}, expected ({self.M.shape[0]}, 3)")

        # Step 2: We need to normalize the normal vectors as the norm of the normal vectors should be 1
        # Hint: You can use function normalize from sklearn.preprocessing
        # Step 2: 归一化法线向量，使其范数为1
        self.N = normalize(self.N, axis=1)
        print("After normalize. N shape:", self.N.shape)

        if self.background_ind is not None:
            for i in range(self.N.shape[1]):
                self.N[self.background_ind, i] = 0


        #############################################





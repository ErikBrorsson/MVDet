import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
import json
from torchvision.datasets import VisionDataset

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']


class MultiviewX(VisionDataset):
    def __init__(self, root, cameras=None):#[1,2,3,4,5,6]):
        super().__init__(root)

        self.root = root
        self.gt_fname = os.path.join(self.root,'gt.txt')
        self.__name__ = 'MultiviewX'
        with open(os.path.join(self.root,'config.json'), 'r') as f:
            config = json.load(f)
        self.num_cam, self.num_frames = config['num_cam'], config['num_frames']
        self.dataset_name = config['Dataset']
        self.img_shape, self.world_grid_shape = config['img_shape'], config['grid_shape']
        self.grid_cell, self.origin = config['grid_cell'], config['origin']
        self.region_size = config['region_size'] 
        self.indexing = 'xy'
        self.worldgrid2worldcoord_mat = np.array([[0,self.grid_cell, self.origin[0]], [self.grid_cell, 0, self.origin[1]], [0, 0, 1]])

        if cameras is not None:
            self.cameras = [x - 1 for x in cameras] # in the code, the camera index is sometimes used to reference the position in a list => need range 0-6 instead of 1-7
            self.num_cam = len(self.cameras)
        else:
            self.cameras = [x for x in range(self.num_cam)]

        self.intrinsic_matrices, self.extrinsic_matrices = {}, {}
        for cam in self.cameras:
            self.intrinsic_matrices[cam], self.extrinsic_matrices[cam] = self.get_intrinsic_extrinsic_matrix(cam)

        print(self.root)
        print(f'Dataset Name : {self.dataset_name}')
        print(f'Cameras : {self.num_cam}, Frames : {self.num_frames}')
        print(f'Image Shape(H,W) : {self.img_shape}')
        print(f'Grid Shape(rows,cols) : {self.world_grid_shape}')
        print(f'Grid Cell(in cm) : {self.grid_cell}cm i.e {self.grid_cell/100}m')
        print(f'Grid Origin(x,y) : {self.origin}')
        print(f'Area/Region size(in m) : {self.region_size[0]}m x {self.region_size[1]}m')
        
        
    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in self.cameras}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam not in self.cameras:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 1000

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord
        grid_x = coord_x * 40
        grid_y = coord_y * 40
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = grid_x / 40
        coord_y = grid_y / 40
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                      intrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                      extrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam


def test():
    from multiview_detector.utils.projection import get_imagecoord_from_worldcoord
    dataset = MultiviewX(os.path.expanduser('~/Data/MultiviewX'), )
    pom = dataset.read_pom()

    foot_3ds = dataset.get_worldcoord_from_pos(np.arange(np.product(dataset.worldgrid_shape)))
    errors = []
    for cam in dataset.cameras:
        projected_foot_2d = get_imagecoord_from_worldcoord(foot_3ds, dataset.intrinsic_matrices[cam],
                                                           dataset.extrinsic_matrices[cam])
        for pos in range(np.product(dataset.worldgrid_shape)):
            bbox = pom[pos][cam]
            foot_3d = dataset.get_worldcoord_from_pos(pos)
            if bbox is None:
                continue
            foot_2d = [(bbox[0] + bbox[2]) / 2, bbox[3]]
            p_foot_2d = projected_foot_2d[:, pos]
            p_foot_2d = np.maximum(p_foot_2d, 0)
            p_foot_2d = np.minimum(p_foot_2d, [1920, 1080])
            errors.append(np.linalg.norm(p_foot_2d - foot_2d))

    print(f'average error in image pixels: {np.average(errors)}')
    pass


if __name__ == '__main__':
    test()

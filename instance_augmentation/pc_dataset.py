# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py 

import os
from xml.etree.ElementTree import TreeBuilder
import numpy as np
import teaserpp_python
from torch.utils import data
import yaml
import pickle
import errno
#import open3d as o3d
#import pygicp
#from probreg import cpd
#import open3d as o3

import copy
from tqdm import tqdm
from sklearn.cluster import KMeans

REGISTERED_PC_DATASET_CLASSES = {}

# solver_params = teaserpp_python.RobustRegistrationSolver.Params()
# solver_params.cbar2 = 1
# solver_params.noise_bound = 0.03
# solver_params.estimate_scaling = False
# solver_params.rotation_estimation_algorithm = (
#     teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
# )
# solver_params.rotation_gnc_factor = 1.4
# solver_params.rotation_max_iterations = 50
# solver_params.rotation_cost_threshold = 1e-1
# solver_params = solver_params
# #print("TEASER++ Parameters are:", solver_params)
# #teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)
# solver = teaserpp_python.RobustRegistrationSolver(solver_params)

# def solve_r(src, target):
#     solver.solve(src, dst)
#     solution = solver.getSolution()
#     solver.reset(solver_params)
#     print("* " * 10)

# for ii in range(10):
#     src = np.load("src.npy").astype(np.float64)
#     dst = np.load("target.npy").astype(np.float64)
#     solve_r(src, dst)
    

def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]

@register_dataset
class SemKITTI_demo(data.Dataset):
    def __init__(self, data_path, imageset='demo',
                 return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.return_ref = return_ref

        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.label_idx = []
        if self.imageset == 'val':
            print(demo_label_path)
            self.label_idx += absoluteFilePaths(demo_label_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'demo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == 'val':
            annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label

from os.path import join
@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train',method='offset',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        thing_class = semkittiyaml['thing_class']
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]
        self.instance_pkl_path = "/dev/JF/instance_aug/"+method+"/sequences/instance_save_path.pkl"
        with open(self.instance_pkl_path, 'rb') as f:
            self.multi_instance_path = pickle.load(f)
        self.CLS_LOSS_WEIGHT = np.ones((20,), dtype=np.float64)  # TODO fix this with content
        self.imageset = imageset
        self.save_cnt = 0
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
            # split = [1]
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        multiscan = 4 # additional two frames are fused with target-frame. Hence, 3 point clouds in total
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

        with open("/data1/SemanticKITTI/dataset/sequences/instance/instance_path.pkl", 'rb') as f:
            self.instance_path = pickle.load(f)

        with open("/data1/SemanticKITTI/dataset/sequences/instance/instance_label.pkl", 'rb') as f:
            self.instance_label = pickle.load(f)

        # solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        # solver_params.cbar2 = 1
        # solver_params.noise_bound = 0.03
        # solver_params.estimate_scaling = False
        # solver_params.rotation_estimation_algorithm = (
        #     teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        # )
        # solver_params.rotation_gnc_factor = 1.4
        # solver_params.rotation_max_iterations = 50
        # solver_params.rotation_cost_threshold = 1e-1
        # self.solver_params = solver_params
        # #print("TEASER++ Parameters are:", solver_params)
        # #teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

        # self.solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)

        # self.masks = np.random.rand(100000)

        # src = np.load("src.npy").astype(np.float64)
        # dst = np.load("target.npy").astype(np.float64)
        # for ii in range(10):
        #     '''
        #     src = np.random.rand(3, 150)
        #     # Apply arbitrary scale, translation and rotation
        #     scale = 1.5
        #     translation = np.array([[100], [0], [-100]])
        #     rotation = np.array([[0.98370992, 0.17903344,    -0.01618098],
        #              [-0.04165862, 0.13947877,    -0.98934839],
        #              [-0.17486954, 0.9739059,    0.14466493]])
        #     dst = scale * np.matmul(rotation, src) + translation
        #     '''
        #     # Add two outliers
        #     #dst[:, 1] += 100
        #     #dst[:, 2] += 15
        #     solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        #     solver.solve(src, dst)
        #     solution = solver.getSolution()
        #     #solver.reset(solver_params)
        #     print("* " * 10)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        # pose = poses[0][idx]

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        # raw_data[:,:3].tofile('1.bin')
        raw_data.tofile('/data2/QSM/CYLIDER3D/Cylinder3D-master/example/{}_raw.bin'.format(index))
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            inst_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            inst_data = annotated_data.copy()
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        if self.imageset == 'val':
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
            if self.return_ref:
                data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan

            return data_tuple
        label = np.vectorize(self.learning_map.__getitem__)(annotated_data)
        # label.astype(np.uint32).tofile('1.label')
        label.astype(np.int32).tofile('/data2/QSM/CYLIDER3D/Cylinder3D-master/example/{}_raw.label'.format(index))

        # ########################################## load saved file ###################################################
        # multi_instance_dict = {}
        # multi_label_dict = {}
        # multi_instance_path = "/dev/JF/instance_aug/sequences/" + str(self.im_idx[index][-22:-20]) +\
        #                         "/instance/" + self.im_idx[index][-10:-4]
        # try:
        #     for multi_inst_path in self.multi_instance_path[multi_instance_path]:
        #         multi_tmp_path = multi_inst_path + '_' + str(multi_inst_path) + '.bin'
        #         multi_instance_dict[multi_inst_path] = np.fromfile(multi_tmp_path, dtype=np.float32).reshape((-1, 4))
        #         multi_label_dict[inst_path] = np.array(self.multi_instance_label[multi_tmp_path]).repeat(multi_instance_dict[multi_inst_path].shape[0], 0)
        # except:
        #     pass
        # multi_list_all = []
        # for multi_key in multi_instance_dict:        
        #     multi_list_all.append( multi_instance_dict[multi_key] ) 
        # if len(multi_list_all) > 0:
        #     multi_list_all = np.concatenate(multi_list_all, 0)
        #     # raw_data = np.concatenate((raw_data, list_all), 0)

        # multi_anno_all = []
        # multi_inst_all = []
        # for multi_key in multi_label_dict:
        #     multi_anno_all.append(multi_label_dict[multi_key])
        #     multi_inst_all.append(np.ones_like(multi_label_dict[multi_key], dtype=multi_label_dict[multi_key].dtype)*multi_key)
        # if len(multi_anno_all) > 0:    
        #     multi_anno_all = np.concatenate(multi_anno_all, 0).reshape(-1,1)
        #     multi_inst_all = np.concatenate(multi_inst_all, 0).reshape(-1,1)
        #     # annotated_data = np.concatenate((annotated_data, anno_all), 0)
        #     # inst_data = np.concatenate((inst_data, inst_all), 0)
        ############################################ load saved file ###################################################
        ########################################## load saved file with self.instance_path #############################
        multi_instance_dict = {}
        multi_label_dict = {}
        instance_path = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + str(self.im_idx[index][-22:-20]) + "/instance/" + self.im_idx[index][-10:-4]
        multi_instance_path = "/dev/JF/instance_aug/registration/sequences/" + str(self.im_idx[index][-22:-20]) +\
                                "/instance/" + self.im_idx[index][-10:-4]
        try:
            for multi_inst_path in self.instance_path[instance_path]:
                tmp_path = instance_path + '_' + str(multi_inst_path) + '.bin'
                if self.instance_label[tmp_path] == 1:
                    break
                multi_label_dict[multi_inst_path] = np.array(self.instance_label[tmp_path]).repeat(multi_instance_dict[multi_inst_path].shape[0], 0)
                multi_tmp_path = multi_instance_path + '_' + str(multi_inst_path) + '.bin'
                multi_instance_dict[multi_inst_path] = np.fromfile(multi_tmp_path, dtype=np.float32).reshape((-1, 4))
                
        except:
            pass
        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
        
        multi_list_all = []
        for multi_key in multi_instance_dict:        
            multi_list_all.append( multi_instance_dict[multi_key] ) 
        if len(multi_list_all) > 0:
            multi_list_all = np.concatenate(multi_list_all, 0)
            raw_data = np.concatenate((raw_data, multi_list_all), 0)

        multi_anno_all = []
        multi_inst_all = []
        for multi_key in multi_label_dict:
            multi_anno_all.append(multi_label_dict[multi_key])
            multi_inst_all.append(np.ones_like(multi_label_dict[multi_key], dtype=multi_label_dict[multi_key].dtype)*multi_key)
        if len(multi_anno_all) > 0:    
            multi_anno_all = np.concatenate(multi_anno_all, 0).reshape(-1,1)
            multi_inst_all = np.concatenate(multi_inst_all, 0).reshape(-1,1)
            annotated_data = np.concatenate((annotated_data, multi_anno_all), 0)
            inst_data = np.concatenate((inst_data, multi_inst_all), 0)
        # multi_list_all[:,:3].tofile('2.bin')
        # multi_anno_all.astype(np.uint32).tofile('2.label')
        raw_data.tofile('/data2/QSM/CYLIDER3D/Cylinder3D-master/example/{}_multi.bin'.format(index))
        annotated_data.astype(np.int32).tofile('/data2/QSM/CYLIDER3D/Cylinder3D-master/example/{}_multi.label'.format(index))
        ########################################## load saved file ###################################################
        # instance_dict = {}
        # label_dict = {}
        # #print(self.im_idx[index], "^ " * 10)
        # path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + str(self.im_idx[index][-22:-20]) + "/instance/" + self.im_idx[index][-10:-4]
        # try:
        #     for inst_path in self.instance_path[path_key]:
        #         tmp_path = path_key + '_' + str(inst_path)+'.bin'
        #         instance_dict[inst_path] = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))
        #         label_dict[inst_path] = np.array(self.instance_label[tmp_path]).repeat(instance_dict[inst_path].shape[0], 0)
        # except:
        #     pass

        # number_idx = int(self.im_idx[index][-10:-4])
        # dir_idx = int(self.im_idx[index][-22:-20])
        # pose0 = self.poses[dir_idx][number_idx]


        # if number_idx - self.multiscan >= 0:
        #     muti_num = np.random.randint(self.multiscan, self.multiscan+1)
        #     for fuse_idx in range(muti_num):
        #         plus_idx = fuse_idx + 1
        #         pose = self.poses[dir_idx][number_idx - plus_idx]

        #         newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
        #         raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

        #         path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + str(self.im_idx[index][-22:-20]) + "/instance/" + newpath2[-10:-4]
        #         try:
        #             for inst_path in self.instance_path[path_key]:
        #                 if not (inst_path in instance_dict):
        #                     continue
        #                 tmp_path = path_key + '_' + str(inst_path)+'.bin'
        #                 tmp_point = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))

        #                 if True or (tmp_point.shape[0] < 5) or (instance_dict[inst_path].shape[0] < 5) or (tmp_point.shape[0] > 2500) or (instance_dict[inst_path].shape[0] > 5000):
        #                     offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
        #                     tmp_point[:, :3] += offest.reshape(1,3)
        #                 else:
        #                     '''
        #                     flag = 0
        #                     if tmp_point[:,:3].shape[0] > 100:
        #                         mask = np.random.rand(tmp_point[:,:3].shape[0])
        #                         thresh = mask > 100. / tmp_point[:,:3].shape[0]
        #                         flag += 1
                            
        #                     if instance_dict[inst_path][:,:3].shape[0] > 300:
        #                         mask2 = np.random.rand(instance_dict[inst_path][:,:3].shape[0])
        #                         thresh2 = mask > 300. / tmp_point[:,:3].shape[0]
        #                         flag += 1
        #                     if flag > 1:
        #                         self.solver.solve(tmp_point[:,:3][~thresh,:].T, instance_dict[inst_path][:,:3][~thresh2,:].T)
        #                     else:
        #                         self.solver.solve(tmp_point[:,:3].T, instance_dict[inst_path][:,:3].T)
        #                     '''
        #                     try:
        #                         #print(tmp_point[:,:3].T.shape, instance_dict[inst_path][:,:3].T.shape, "F  A")
        #                         src = copy.deepcopy(tmp_point[:,:3].T)
        #                         target = copy.deepcopy(instance_dict[inst_path][:,:3].T)
        #                         #print(src, target, "* " * 10)
        #                         self.solver.reset(self.solver_params)
        #                         self.solver.solve(src, target)
        #                         solution = self.solver.getSolution()
        #                         slu = copy.deepcopy(solution)
        #                         tmp_point[:, :3] = (np.dot(slu.rotation, tmp_point[:,:3].T*slu.scale) + slu.translation.reshape((-1,1))).T
        #                     except:
        #                         offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
        #                         tmp_point[:, :3] += offest.reshape(1,3)
 
        #                 ''' 
        #                 self.gicp.set_input_target(instance_dict[inst_path][:,:3])
        #                 self.gicp.set_input_source(tmp_point[:,:3])               
        #                 matrix = self.gicp.align()
        #                 source2 = np.concatenate([tmp_point[:,:3], np.ones((tmp_point.shape[0],1))], axis=1)
        #                 tmp_point2 = np.dot(matrix[:3,:], source2.transpose(1,0)).transpose(1,0) 
        #                 '''
        #                 '''
        #                 self.target.points = o3.utility.Vector3dVector(instance_dict[inst_path][:,:3])
        #                 self.source.points = o3.utility.Vector3dVector(tmp_point[:,:3])
        #                 tf_param, _, _ = cpd.registration_cpd(self.source, self.target)
        #                 self.source.points = tf_param.transform(self.source.points)
        #                 tmp_point = np.asarray(self.source.points)
        #                 '''
        #                 #tf_param, _, _ = cpd.registration_cpd(tmp_point[:,:3], instance_dict[inst_path][:,:3])
        #                 #tmp_point[:,:3] = tf_param.transform(tmp_point[:,:3])

        #                 #print(np.mean(tmp_point, axis=0), np.mean(instance_dict[inst_path][:,:3], axis=0))
        #                 #tmp_point = np.concatenate([tmp_point2, tmp_point[:,3:4]], axis=1)
        #                 instance_dict[inst_path] = np.concatenate([tmp_point, instance_dict[inst_path]], axis=0) 
        #                 label_dict[inst_path] = np.concatenate([np.array(label_dict[inst_path][0]).repeat(tmp_point.shape[0],0), label_dict[inst_path]], axis=0)
        #         except:
        #             if np.random.rand(1) > 0.99:
        #                 print("error!!!")
        #             pass
 
        #         if self.imageset == 'test':
        #             annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
        #         else:
        #             annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
        #                                           dtype=np.int32).reshape((-1, 1))
        #             annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

        #             ## remove the moving object
        #             #print(np.unique(annotated_data2), label_dict)
        #             '''
        #             mask = np.zeros_like(annotated_data2)
        #             for key in label_dict:
        #                 mask[annotated_data2 == label_dict[key][0]] = 1.
        #             raw_data2 = raw_data2[mask[:,0] < 0.5, :]
        #             annotated_data2 = annotated_data2[mask < 0.5].reshape(-1,1)
        #             '''
        #         '''
        #         raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)
        #         #print(annotated_data.shape, annotated_data2.shape)
        #         if len(raw_data2) != 0:
        #             raw_data = np.concatenate((raw_data, raw_data2), 0)
        #             annotated_data = np.concatenate((annotated_data, annotated_data2), 0)
        #         '''

        # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        # list_all = []
        # for key in instance_dict:        
        #    list_all.append( instance_dict[key] ) 
        # if len(list_all) > 0:
        #     list_all = np.concatenate(list_all, 0)
        #     raw_data = np.concatenate((raw_data, list_all), 0)

        # anno_all = []
        # inst_all = []
        # for key in label_dict:
        #     anno_all.append(label_dict[key])
        #     inst_all.append(np.ones_like(label_dict[key], dtype=label_dict[key].dtype)*key)
        # if len(anno_all) > 0:    
        #     anno_all = np.concatenate(anno_all, 0).reshape(-1,1)
        #     inst_all = np.concatenate(inst_all, 0).reshape(-1,1)
        #     annotated_data = np.concatenate((annotated_data, anno_all), 0)
        #     inst_data = np.concatenate((inst_data, inst_all), 0)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8), inst_data)
        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan

        return data_tuple

    def save_instance(self, out_dir, min_points = 10):
        # this part has problem, please refer to instance_preprocess.py
        print("this part has problem, please refer to instance_preprocess.py")
        instance_dict_save={label:[] for label in self.thing_list}
        instance_path_dict = {}
        print("\033[0;31;40m Attention! registration before save!!!!\033[0m")
        pbar = tqdm(total=len(self.im_idx), ncols=100)
        except_cnt = 0
        for i in range(len(self.im_idx)):
            # print('process instance for:' + self.im_idx[i])
            # get x,y,z,ref,semantic label and instance label
            # this part follow the method of __getitem__(self, index)
            # for easy processing, we just save the data after registration.
            raw_data = np.fromfile(self.im_idx[i], dtype=np.float32).reshape((-1, 4))
            origin_len = len(raw_data)
            if self.imageset == 'test':
                annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            else:
                annotated_data = np.fromfile(self.im_idx[i].replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.int32).reshape((-1, 1))
                annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

            if self.imageset == 'val':
                annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

                data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
                if self.return_ref:
                    data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan

                return data_tuple
            
            instance_dict = {}
            label_dict = {}
            path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + str(self.im_idx[i][-22:-20]) + "/instance/" + self.im_idx[i][-10:-4]
            try:
                for inst_path in self.instance_path[path_key]:
                    tmp_path = path_key + '_' + str(inst_path)+'.bin'
                    instance_dict[inst_path] = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))
                    label_dict[inst_path] = np.array(self.instance_label[tmp_path]).repeat(instance_dict[inst_path].shape[0], 0)
            except:
                pass

            number_idx = int(self.im_idx[i][-10:-4])
            dir_idx = int(self.im_idx[i][-22:-20])
            pose0 = self.poses[dir_idx][number_idx]

            if number_idx - self.multiscan >= 0:
                muti_num = np.random.randint(self.multiscan, self.multiscan+1)
                for fuse_idx in range(muti_num):
                    plus_idx = fuse_idx + 1
                    pose = self.poses[dir_idx][number_idx - plus_idx]

                    newpath2 = self.im_idx[i][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[i][-4:]
                    raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                    path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + \
                        str(self.im_idx[i][-22:-20]) + "/instance/" + newpath2[-10:-4]
                    try:
                        for inst_path in self.instance_path[path_key]:
                            self.solver.reset(self.solver_params)
                            
                            if not (inst_path in instance_dict):
                                continue
                            tmp_path = path_key + '_' + str(inst_path)+'.bin'
                            tmp_point = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))
                            # 这里改为False and 仍然无法进到else分支？
                            if (tmp_point.shape[0] < 10) or (instance_dict[inst_path].shape[0] < 10) or (tmp_point.shape[0] > 2500) or (instance_dict[inst_path].shape[0] > 5000):
                                offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
                                tmp_point[:, :3] += offest.reshape(1,3)
                            else:
                                flag = 0
                                if tmp_point[:,:3].shape[0] > 100:
                                    mask = self.masks[:tmp_point[:,:3].shape[0]] #np.random.rand(tmp_point[:,:3].shape[0])
                                    thresh = mask > 100. / tmp_point[:,:3].shape[0]
                                    flag += 1
                                
                                if instance_dict[inst_path][:,:3].shape[0] > 300:
                                    mask2 = self.masks[:instance_dict[inst_path][:,:3].shape[0]] #np.random.rand(instance_dict[inst_path][:,:3].shape[0])
                                    thresh2 = mask2 > 300. / instance_dict[inst_path][:,:3].shape[0]
                                    flag += 1
                                
                                src = copy.deepcopy(tmp_point[:,:3].T).astype(np.float64)
                                target = copy.deepcopy(instance_dict[inst_path][:,:3].T).astype(np.float64)
                                if flag > 1:
                                    self.solver.solve(src[:,~thresh], target[:,~thresh2])
                                else:
                                    self.solver.solve(src, target)
                                
                                slu = self.solver.getSolution()
                                if (slu.scale > 1.1) or (slu.scale < 0.9):
                                    offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
                                    tmp_point[:, :3] += offest.reshape(1,3)
                                else:
                                    tmp_point[:, :3] = (np.dot(slu.rotation, tmp_point[:,:3].T*slu.scale) + slu.translation.reshape((-1,1))).T
                                #self.solver.reset(self.solver_params)
                                '''
                                try:
                                    #print(tmp_point[:,:3].T.shape, instance_dict[inst_path][:,:3].T.shape, "F  A")
                                    src = copy.deepcopy(tmp_point[:,:3].T).astype(np.float64)
                                    target = copy.deepcopy(instance_dict[inst_path][:,:3].T).astype(np.float64)
                
                                    #src_mean, src_std = np.mean(src, axis=1, keepdims=True), np.std(src, axis=1, keepdims=True)
                                    #np.save("src.npy", src)
                                    #np.save("target.npy", target)
                                
                                    #print(src, target, "* " * 10)
                                    self.solver.reset(self.solver_params)
                                    self.solver.solve(src, target)
                                    solution = self.solver.getSolution()
                                    slu = copy.deepcopy(solution)
                                    
                                    tmp_point[:, :3] = (np.dot(slu.rotation, tmp_point[:,:3].T*slu.scale) + slu.translation.reshape((-1,1))).T
                                except :
                                    offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
                                    tmp_point[:, :3] += offest.reshape(1,3)
                                '''                            
                            instance_dict[inst_path] = np.concatenate([tmp_point, instance_dict[inst_path]], axis=0) 
                            label_dict[inst_path] = np.concatenate([np.array(label_dict[inst_path][0]).repeat(tmp_point.shape[0],0), label_dict[inst_path]], axis=0)
                            #self.solver.reset(self.solver_params)
                    except Exception as e:
                        print(e)
                        if np.random.rand(1) > -0.99:
                            print("error!!!")
                        except_cnt += 1
                        pass
    
                    if self.imageset == 'test':
                        annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                    else:
                        annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                    dtype=np.int32).reshape((-1, 1))
                        annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            # this part is used to save the instance after multi fusion.
            cur_inst_key = []
            for key in instance_dict:
                _,dir2 = self.im_idx[i].split('/sequences/',1)
                new_save_dir = out_dir + '/sequences/' +dir2.replace('velodyne','instance')[:-4]+'_'+str(key)+'.bin'
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                instance_dict[key].tofile(new_save_dir)
                instance_dict_save[int(label_dict[key][0])].append(new_save_dir)
                cur_inst_key.append(key)
            instance_path_dict[out_dir + '/sequences/' +dir2.replace('velodyne','instance')[:-4]] = cur_inst_key
            self.save_cnt += 1
            pbar.update(1)
            # if i == 100:
            #     break
            with open('/dev/JF/instance_aug/registration/sequences'+'/instance_class.pkl', 'wb') as f:
                pickle.dump(instance_dict_save, f)
            with open('/dev/JF/instance_aug/registration/sequences'+'/instance_path.pkl', 'wb') as f:
                pickle.dump(instance_path_dict, f)
        pbar.close()
        return None

    def print_save_cnt(self):
        print('Save {} frames'.format(self.save_cnt))

    def pole_preprocess(self, out_dir, min_points=10):
        save_dict = {'pole_and_traffic':[], 'pole':[], 'traffic': []}
        pbar = tqdm(total=len(self.im_idx))
        for i in range(len(self.im_idx)):
            # i=5383
            raw_data = np.fromfile(self.im_idx[i], dtype=np.float32).reshape((-1, 4))
            annotated_data = np.fromfile(self.im_idx[i].replace('velodyne', 'labels')[:-3] + 'label',
                                                dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            points = raw_data[:,:3]
            sig = raw_data[:,3:]
            if len(sig.shape) == 2: sig=np.squeeze(sig)
            mask = np.zeros_like(annotated_data, dtype=bool)
            for label in [18, 19]:
                mask[annotated_data == label] = True
            if len(mask.shape) == 2: mask = np.squeeze(mask)
            if np.sum(mask) > 10:
                # try:
                annotated_pole = annotated_data[mask]
                points_pole = points[mask]
                sig_pole = sig[mask]
                centroids = self.farthest_point_sample(points_pole, 30, threshold=9)
                n_clusters = centroids.shape[0]
                kmeans = KMeans(n_clusters=n_clusters, init=points_pole[centroids,:]).fit(points_pole)
                instance_label = kmeans.labels_
                unique_label = np.unique(instance_label)
                for inst in unique_label:
                    index = np.where(instance_label == inst)[0]
                    if index.size < 10: continue
                    if np.any(np.var(points_pole[index,:2], axis=0) > 1): continue
                    _, dir2 = self.im_idx[i].split('/sequences',1)
                    pole_save_path = out_dir + '/sequences' + dir2[:-4] + '_' + str(inst) + '.bin'
                    label_save_path = pole_save_path.replace('velodyne', 'label').replace('bin', 'label')
                    if not os.path.exists(os.path.dirname(pole_save_path)):
                        try:
                            os.makedirs(os.path.dirname(pole_save_path))
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise
                    if not os.path.exists(os.path.dirname(label_save_path)):
                        try:
                            os.makedirs(os.path.dirname(label_save_path))
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise
                    point_inst = np.concatenate((points_pole[index,:], sig_pole[index][..., np.newaxis]), axis=1)
                    point_inst.astype(np.float32).tofile(pole_save_path)
                    annotated_pole[index,:].astype(np.int32).tofile(label_save_path)
                    if 19 in annotated_pole[index,:]:
                        if 18 in annotated_pole[index,:]:
                            save_dict['pole_and_traffic'].append(pole_save_path)
                        else:
                            save_dict['traffic'].append(pole_save_path)
                    else:
                        save_dict['pole'].append(pole_save_path)
            pass
            pbar.update(1)
            # with open('/dev/JF/instance_aug/pole/sequences'+'/pole_traffic_path.pkl', 'wb') as f:
            #     pickle.dump(save_dict, f)
        pbar.close()
        return 
    
    def farthest_point_sample(self, xyz, npoint, threshold):
        """
        Input:
            xyz: pointcloud data, [N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint]
        """
        N, C = xyz.shape
        # 初始化一个centroids矩阵，用于存储npoint个采样点的索引位置，大小为B×npoint
        centroids = np.zeros(npoint, dtype=np.int64)
        # distance矩阵(B×N)记录所有点到某一个点的距离，初始化的值很大，后面会迭代更新,随机选择一个点作为固定点
        distance = np.ones(N)* 1e10
        farthest = np.random.randint(0, N, (1,), dtype=np.int64)
        # 直到采样点达到npoint，或者前后两个采样点距离小于阈值，否则进行如下迭代：
        for i in range(npoint):
            # 设当前的采样点centroids为当前的最远点farthest
            centroids[i] = farthest
            # 取出该中心点centroid的坐标
            centroid = xyz[farthest, :].reshape(1, 3)
            # 求出所有点到该centroid点的欧式距离，存在dist矩阵中
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            # 建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值
            # 随着迭代的继续，distance矩阵中的值会慢慢变小，
            # 其相当于记录着每个点距离所有已出现的采样点的最小距离
            mask = dist < distance#确保拿到的是距离所有已选中心点最大的距离。比如已经是中心的点，其dist始终保持为	 #0，二在它附近的点，也始终保持与这个中心点的距离
            distance[mask] = dist[mask]
            # 从distance矩阵取出最远的点为farthest，继续下一轮迭代
            farthest = np.argmax(distance, axis=-1)
            if np.any(np.sum((xyz[centroids[:i],:] - xyz[farthest,:])**2, axis=1) < threshold):
                break
        return centroids[:i+1]



def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name

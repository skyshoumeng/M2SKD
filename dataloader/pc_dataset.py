# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py 

import os
import numpy as np
from torch.utils import data
import yaml
import pickle
#import open3d as o3d
#import pygicp
#from probreg import cpd
#import open3d as o3
import teaserpp_python
import copy

REGISTERED_PC_DATASET_CLASSES = {}

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


@register_dataset
class SemKITTI_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)
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
    def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
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

        #self.gicp = pygicp.FastGICP()
        #self.target = o3.geometry.PointCloud()
        #self.source = o3.geometry.PointCloud()

        with open("/data1/SemanticKITTI/dataset/sequences/instance/instance_path.pkl", 'rb') as f:
            self.instance_path = pickle.load(f)

        with open("/data1/SemanticKITTI/dataset/sequences/instance/instance_label.pkl", 'rb') as f:
            self.instance_label = pickle.load(f)

        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 0.03
        solver_params.estimate_scaling = True
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 50
        solver_params.rotation_cost_threshold = 1e-12
        self.solver_params = solver_params
        #print("TEASER++ Parameters are:", solver_params)
        #teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

        self.solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)


        for ii in range(100):
            src = np.random.rand(3, 150)
            # Apply arbitrary scale, translation and rotation
            scale = 1.5
            translation = np.array([[100], [0], [-100]])
            rotation = np.array([[0.98370992, 0.17903344,    -0.01618098],
                     [-0.04165862, 0.13947877,    -0.98934839],
                     [-0.17486954, 0.9739059,    0.14466493]])
            dst = scale * np.matmul(rotation, src) + translation

            # Add two outliers
            dst[:, 1] += 100
            dst[:, 2] += 15

            self.solver.solve(src, dst)
            self.solution = self.solver.getSolution()
            self.solver.reset(self.solver_params)


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
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        if self.imageset == 'val':
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
            if self.return_ref:
                data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan

            return data_tuple


        instance_dict = {}
        label_dict = {}
        #print(self.im_idx[index], "^ " * 10)
        path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + str(self.im_idx[index][-22:-20]) + "/instance/" + self.im_idx[index][-10:-4]
        try:
            for inst_path in self.instance_path[path_key]:
                tmp_path = path_key + '_' + str(inst_path)+'.bin'
                instance_dict[inst_path] = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))
                label_dict[inst_path] = np.array(self.instance_label[tmp_path]).repeat(instance_dict[inst_path].shape[0], 0)
        except:
            pass

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])
        pose0 = self.poses[dir_idx][number_idx]

        if number_idx - self.multiscan >= 0:
            muti_num = np.random.randint(self.multiscan, self.multiscan+1)
            for fuse_idx in range(muti_num):
                plus_idx = fuse_idx + 1
                pose = self.poses[dir_idx][number_idx - plus_idx]

                newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + str(self.im_idx[index][-22:-20]) + "/instance/" + newpath2[-10:-4]
                try:
                    for inst_path in self.instance_path[path_key]:
                        if not (inst_path in instance_dict):
                            continue
                        if  label_dict[inst_path][0] == 1:
                            continue
                        tmp_path = path_key + '_' + str(inst_path)+'.bin'
                        tmp_point = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))

                        if True or (tmp_point.shape[0] < 5) or (instance_dict[inst_path].shape[0] < 5) or (tmp_point.shape[0] > 2500) or (instance_dict[inst_path].shape[0] > 5000):
                            offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
                            tmp_point[:, :3] += offest.reshape(1,3)
                        else:
                            '''
                            flag = 0
                            if tmp_point[:,:3].shape[0] > 100:
                                mask = np.random.rand(tmp_point[:,:3].shape[0])
                                thresh = mask > 100. / tmp_point[:,:3].shape[0]
                                flag += 1
                            
                            if instance_dict[inst_path][:,:3].shape[0] > 300:
                                mask2 = np.random.rand(instance_dict[inst_path][:,:3].shape[0])
                                thresh2 = mask > 300. / tmp_point[:,:3].shape[0]
                                flag += 1
                            if flag > 1:
                                self.solver.solve(tmp_point[:,:3][~thresh,:].T, instance_dict[inst_path][:,:3][~thresh2,:].T)
                            else:
                                self.solver.solve(tmp_point[:,:3].T, instance_dict[inst_path][:,:3].T)
                            '''
                            try:
                                #print(tmp_point[:,:3].T.shape, instance_dict[inst_path][:,:3].T.shape, "F  A")
                                src = copy.deepcopy(tmp_point[:,:3].T)
                                target = copy.deepcopy(instance_dict[inst_path][:,:3].T)
                                #print(src, target, "* " * 10)
                                self.solver.reset(self.solver_params)
                                self.solver.solve(src, target)
                                solution = self.solver.getSolution()
                                slu = copy.deepcopy(solution)
                                tmp_point[:, :3] = (np.dot(slu.rotation, tmp_point[:,:3].T*slu.scale) + slu.translation.reshape((-1,1))).T
                            except:
                                offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
                                tmp_point[:, :3] += offest.reshape(1,3)
 
                        ''' 
                        self.gicp.set_input_target(instance_dict[inst_path][:,:3])
                        self.gicp.set_input_source(tmp_point[:,:3])               
                        matrix = self.gicp.align()
                        source2 = np.concatenate([tmp_point[:,:3], np.ones((tmp_point.shape[0],1))], axis=1)
                        tmp_point2 = np.dot(matrix[:3,:], source2.transpose(1,0)).transpose(1,0) 
                        '''
                        '''
                        self.target.points = o3.utility.Vector3dVector(instance_dict[inst_path][:,:3])
                        self.source.points = o3.utility.Vector3dVector(tmp_point[:,:3])
                        tf_param, _, _ = cpd.registration_cpd(self.source, self.target)
                        self.source.points = tf_param.transform(self.source.points)
                        tmp_point = np.asarray(self.source.points)
                        '''
                        #tf_param, _, _ = cpd.registration_cpd(tmp_point[:,:3], instance_dict[inst_path][:,:3])
                        #tmp_point[:,:3] = tf_param.transform(tmp_point[:,:3])

                        #print(np.mean(tmp_point, axis=0), np.mean(instance_dict[inst_path][:,:3], axis=0))
                        #tmp_point = np.concatenate([tmp_point2, tmp_point[:,3:4]], axis=1)
                        instance_dict[inst_path] = np.concatenate([tmp_point, instance_dict[inst_path]], axis=0) 
                        label_dict[inst_path] = np.concatenate([np.array(label_dict[inst_path][0]).repeat(tmp_point.shape[0],0), label_dict[inst_path]], axis=0)
                except:
                    if np.random.rand(1) > 0.99:
                        print("error!!!")
                    pass
 
                if self.imageset == 'test':
                    annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                else:
                    annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                  dtype=np.int32).reshape((-1, 1))
                    annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

                    ## remove the moving object
                    #print(np.unique(annotated_data2), label_dict)
                    '''
                    mask = np.zeros_like(annotated_data2)
                    for key in label_dict:
                        mask[annotated_data2 == label_dict[key][0]] = 1.
                    raw_data2 = raw_data2[mask[:,0] < 0.5, :]
                    annotated_data2 = annotated_data2[mask < 0.5].reshape(-1,1)
                    '''
                '''
                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)
                #print(annotated_data.shape, annotated_data2.shape)
                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    annotated_data = np.concatenate((annotated_data, annotated_data2), 0)
                '''

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        list_all = []
        for key in instance_dict:        
           list_all.append( instance_dict[key] ) 
        if len(list_all) > 0:
            list_all = np.concatenate(list_all, 0)
            raw_data = np.concatenate((raw_data, list_all), 0)
        
        anno_all = []
        for key in label_dict:
            anno_all.append( label_dict[key] )
        if len(anno_all) > 0:    
            anno_all = np.concatenate(anno_all, 0).reshape(-1,1)
            annotated_data = np.concatenate((annotated_data, anno_all), 0)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan


        return data_tuple


# load Semantic KITTI class info

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

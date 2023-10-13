#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from xml.sax.saxutils import prepare_input_source
GRANDFA = os.path.dirname(os.path.realpath(__file__))
sys.path.append(GRANDFA)
# from torch._C import T
import teaserpp_python
import numpy as np
import yaml
import pickle
from tqdm import tqdm
import copy
import errno
import time


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

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
solver_params = solver_params
print("TEASER++ Parameters are:", solver_params)
#teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)
solver = teaserpp_python.RobustRegistrationSolver(solver_params)

def solve_r(src, target):
    solver.reset(solver_params)
    solver.solve(src, dst)
    solution = solver.getSolution()
    
    return solution

# for ii in range(10):
#     src = np.load("/data2/QSM/CYLIDER3D/Cylinder3D-master/instance_augmentation/src.npy").astype(np.float64)
#     dst = np.load("/data2/QSM/CYLIDER3D/Cylinder3D-master/instance_augmentation/target.npy").astype(np.float64)
#     slu = solve_r(src, dst)
#     result = (np.dot(slu.rotation, src[:,:3].T*slu.scale) +\
#                                 slu.translation.reshape((-1,1))).T
#     #print(slu, "M " * 10)
# if ii == 9:
#     print("finish test!")

#####################################
print("\033[0;31;40m Attention! Use registration for multi fusion!!!!\033[0m")
data_path = '/data1/SemanticKITTI/dataset' + '/sequences/'
out_path = '/dev/JF/instance_aug/' 
method = 'registration'
assert method in ['offset', 'registration']
out_path = out_path + method + '/'
print("\033[0;31;40m result after multi-fusion save in {}!!!!\033[0m".format(out_path))

label_mapping="/data2/QSM/CYLIDER3D/Cylinder3D-master/instance_augmentation/semantic-kitti.yaml"
with open(label_mapping, 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']
thing_class = semkittiyaml['thing_class']
thing_list = [cl for cl, ignored in thing_class.items() if ignored]
split = semkittiyaml['split']['train']

multiscan = 4

im_idx = []
for i_folder in split:
    im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))
with open("/data1/SemanticKITTI/dataset/sequences/instance/instance_path.pkl", 'rb') as f:
    instance_path = pickle.load(f)

with open("/data1/SemanticKITTI/dataset/sequences/instance/instance_label.pkl", 'rb') as f:
    instance_label = pickle.load(f)
masks = np.random.rand(100000)

instance_dict_save={label:[] for label in thing_list}
instance_path_dict = {}
pbar = tqdm(total=len(im_idx), ncols=100)
for i in range(len(im_idx)):
    raw_data = np.fromfile(im_idx[i], dtype=np.float32).reshape((-1, 4))
    annotated_data = np.fromfile(im_idx[i].replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.int32).reshape((-1, 1))
    annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
    
    instance_dict = {}
    label_dict = {}
    first_len_dict = {}
    path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" +\
                str(im_idx[i][-22:-20]) + "/instance/" + im_idx[i][-10:-4]
    try:
        for inst_path in instance_path[path_key]:
            tmp_path = path_key + '_' + str(inst_path)+'.bin'
            instance_dict[inst_path] = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))
            first_len_dict[inst_path] = instance_dict[inst_path].shape[0]
            label_dict[inst_path] = np.array(instance_label[tmp_path]).repeat(instance_dict[inst_path].shape[0], 0)
    except:
        pass

    
    number_idx = int(im_idx[i][-10:-4])  # frame number
    dir_idx = int(im_idx[i][-22:-20])  # sequences number
    if number_idx - multiscan >= 0:
        
        muti_num = np.random.randint(multiscan, multiscan+1)
        
        for fuse_idx in range(muti_num):
            plus_idx = fuse_idx + 1
            newpath2 = im_idx[i][:-10] + str(number_idx - plus_idx).zfill(6) + im_idx[i][-4:]
            raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))
            path_key = "/data1/SemanticKITTI/dataset/sequences/instance/sequences/" + \
                        str(im_idx[i][-22:-20]) + "/instance/" + newpath2[-10:-4]
            
            try:                    
                for inst_path in instance_path[path_key]:
                    if not (inst_path in instance_dict):
                        continue
                    
                    first_len = first_len_dict[inst_path]
                    
                    tmp_path = path_key + '_' + str(inst_path)+'.bin'
                    tmp_point = np.fromfile(tmp_path, dtype=np.float32).reshape((-1, 4))
                    if (tmp_point.shape[0] < 20) or (instance_dict[inst_path][:first_len,:].shape[0] < 20) or (tmp_point.shape[0] > 2500) or (instance_dict[inst_path][:first_len,:].shape[0] > 5000):
                        offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:,:3], axis=0)
                        tmp_point[:, :3] += offest.reshape(1,3)
                    else:
                        flag = 0
                        if tmp_point[:,:3].shape[0] > 200:
                            mask = masks[:tmp_point[:,:3].shape[0]] #np.random.rand(tmp_point[:,:3].shape[0])
                            thresh = mask > 200. / tmp_point[:,:3].shape[0]
                            flag += 1
                        
                        if instance_dict[inst_path][:first_len,:][:,:3].shape[0] > 500:
                            mask2 = masks[:instance_dict[inst_path][:first_len,:][:,:3].shape[0]] #np.random.rand(instance_dict[inst_path][:,:3].shape[0])
                            thresh2 = mask2 > 500. / instance_dict[inst_path][:,:3].shape[0]
                            flag += 1
                        
                        src = copy.deepcopy(tmp_point[:,:3])#.astype(np.float64)
                        target = copy.deepcopy(instance_dict[inst_path][:first_len,:][:,:3])#.astype(np.float64)
                        # np.save("src.npy", src)
                        # np.save("target.npy", target)
                        # break
                        
                        # if np.mean(target, axis=1)[2] > 8:
                        # #     np.save("src.npy", src)
                        # #     np.save("target.npy", target)
                        #     print(np.mean(src,axis=1), np.mean(target, axis=1))
                        #     print(im_idx[i], "\n", inst_path, '\n', tmp_path)
                        #     raise Exception("saved !!!")
                        if flag > 1:
                            src = src[~thresh,:]
                            target = target[~thresh2,:]
                        
                        
                        src_m, target_m = np.mean(src, axis=0), np.mean(target, axis=0)
                        src = src - target_m
                        target = target - target_m
                        
                        if src.shape[0] > target.shape[0]:
                            #np.random.shuffle(src)
                            #index_t = np.random.randint(0, src.shape[0], target.shape[0])
                            src = src[:target.shape[0], :]
                        else:
                            #print(src.shape, target.shape, "* " * 10)
                            #np.random.shuffle(target)
                            #index_t = np.random.randint(0, target.shape[0], src.shape[0])
                            target = target[:src.shape[0], :]
                        
                        #print(np.mean(src.T,axis=1), np.mean(target.T, axis=1), target.shape)
                        #print(src.shape, target.shape)
                        use_w = True
                        for tr in range(1):
                            solver.reset(solver_params)
                            solver.solve(src.T, target.T)
                            slu = solver.getSolution()
                            if slu.rotation[0,0] > 0.8 and slu.rotation[1,1] > 0.8 and slu.rotation[2,2] > 0.8 and np.sum(np.abs(slu.translation+np.mean(src,axis=0))) < .8:
                                use_w = False
                                break
                            
                        #print(slu.translation.shape, target_m.shape, slu.translation.shape, np.mean(src,axis=0).shape)
                        
                        if (slu.scale > 1.3) or (slu.scale < 0.7) or use_w:
                            offest = -np.mean(tmp_point[:, :3], axis=0) + np.mean(instance_dict[inst_path][:first_len,:][:,:3], axis=0)
                            tmp_point[:, :3] += offest.reshape(1,3)
                        else:
                            #print("* " * 10)
                            tmp_point[:, :3] = (np.dot(slu.rotation, (tmp_point[:,:3] - target_m).T*slu.scale) + slu.translation.reshape((-1,1))).T + target_m
                        #print(slu,  src.shape, np.mean(tmp_point[:,:3], axis=0), "*"*10)
                        #print(np.dot(tmp_point[:,:3]*slu.scale, slu.rotation).shape, slu.translation.reshape((-1,1)).shape, "^ " * 10)
                        # break
                    instance_dict[inst_path] = np.concatenate([instance_dict[inst_path], tmp_point], axis=0) 
                    label_dict[inst_path] = np.concatenate([label_dict[inst_path], 
                                                            np.array(label_dict[inst_path][0]).repeat(tmp_point.shape[0],0)], axis=0)    
            except Exception as e:
                print(e)
                if np.random.rand(1) > -0.99:
                    print("error!!!")
            # break            
    annotated_data = np.vectorize(learning_map.__getitem__)(annotated_data)
    # this part is used to save the instance after multi fusion.
    cur_inst_key = []
    for key in instance_dict:
        _,dir2 = im_idx[i].split('/sequences/',1)
        new_save_dir = out_path + '/sequences/' +dir2.replace('velodyne','instance')[:-4]+'_'+str(key)+'.bin'
        if not os.path.exists(os.path.dirname(new_save_dir)):
            try:
                os.makedirs(os.path.dirname(new_save_dir))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        instance_dict[key].tofile(new_save_dir)
        instance_dict_save[int(label_dict[key][0])].append(new_save_dir)
        cur_inst_key.append(key)
    instance_path_dict[out_path + '/sequences/' +dir2.replace('velodyne','instance')[:-4]] = cur_inst_key
    with open('/dev/JF/instance_aug/registration/sequences'+'/instance_class.pkl', 'wb') as f:
        pickle.dump(instance_dict_save, f)
    with open('/dev/JF/instance_aug/registration/sequences'+'/instance_path.pkl', 'wb') as f:
        pickle.dump(instance_path_dict, f)
    pbar.update(1)
    # break
pbar.close()
#####################################
print('instance preprocessing finished.')


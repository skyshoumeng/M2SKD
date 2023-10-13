#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle

class instance_augmentation(object):
    def __init__(self,instance_path,pole_instance_dict, thing_list,class_weight,random_flip = False,
                random_add = False,random_rotate = False,local_transformation = False):
        self.thing_list = thing_list
        self.thing_list = [2,3,4,5,6,7,8]
        # class_weight
        self.instance_weight = [class_weight[thing_class_num-1] for thing_class_num in self.thing_list]
        self.instance_weight = np.asarray(self.instance_weight)/np.sum(self.instance_weight)
        self.random_flip = random_flip
        self.random_add = random_add
        self.random_rotate = random_rotate
        self.local_transformation = local_transformation

        self.add_num = 10
        self.instance_path = instance_path
        self.pole_instance_dict = pole_instance_dict

    def instance_aug(self, point_xyz, point_label, point_inst, point_feat = None, org_len=None):
        """random rotate and flip each instance independently.

        Args:
            point_xyz: [N, 3], point location
            point_label: [N, 1], class label
            point_inst: [N, 1], instance label
        """        
        # random add instance to this scan
        assert org_len is not None, print('We need original length for process.')
        if self.random_add:
            # choose which instance to add
            instance_choice = np.random.choice(len(self.thing_list),self.add_num,replace=True,p=self.instance_weight)
            uni_inst, uni_inst_count = np.unique(instance_choice,return_counts=True)
            add_idx = 1
            total_point_num = 0
            early_break = False
            for n, count in zip(uni_inst, uni_inst_count):
                # find random instance
                random_choice = np.random.choice(len(self.instance_path[self.thing_list[n]]),count)
                # add to current scan
                for idx in random_choice:
                    # 加载多帧
                    points_multi = np.fromfile(self.instance_path[self.thing_list[n]][idx], dtype=np.float32).reshape((-1, 4))
                    add_xyz_multi = points_multi[:,:3]
                    center = np.mean(add_xyz_multi,axis=0)
                    # 对应的单帧
                    points_path = self.instance_path[self.thing_list[n]][idx]
                    points_path = '/data1/SemanticKITTI/dataset/sequences/instance/sequences'+\
                        points_path.split('sequences',1)[1]
                    points = np.fromfile(points_path, dtype=np.float32).reshape((-1, 4))
                    add_xyz = points[:,:3]

                    # need to check occlusion
                    fail_flag = True
                    if self.random_rotate:
                        # random rotate
                        random_choice = np.random.random(20)*np.pi*2
                        for r in random_choice:
                            center_r = self.rotate_origin(center[np.newaxis,...],r)
                            # check if occluded
                            if self.check_occlusion(point_xyz,center_r[0], point_label):
                                fail_flag = False
                                break
                        # rotate to empty space
                        if fail_flag: continue
                        add_xyz = self.rotate_origin(add_xyz,r)
                        add_xyz_multi = self.rotate_origin(add_xyz_multi,r)
                    else:
                        fail_flag = not self.check_occlusion(point_xyz,center, point_label)
                    if fail_flag: continue

                    # 对应的标签，instance标签
                    add_label = np.ones((points.shape[0],),dtype=np.uint8)*(self.thing_list[n])
                    add_label_multi = np.ones((points_multi.shape[0],),dtype=np.uint8)*(self.thing_list[n])

                    add_inst = np.ones((points.shape[0],),dtype=np.uint32)*(add_idx<<16)
                    add_inst_multi = np.ones((points_multi.shape[0],),dtype=np.uint32)*(add_idx<<16)
                    
                    # 拼接，注意单帧应当拼接在中间！
                    point_xyz1 = np.concatenate((point_xyz[:org_len,:],add_xyz),axis=0)
                    point_label1 = np.concatenate((point_label[:org_len],add_label),axis=0)
                    point_inst1 = np.concatenate((point_inst[:org_len],add_inst),axis=0)
                    org_len1 = point_xyz1.shape[0]
                    # point_xyz1.tofile('6.bin')
                    # point_label1.tofile('6.label')
                    point_xyz = np.concatenate((point_xyz1, point_xyz[org_len:,:],add_xyz_multi),axis=0)
                    point_label = np.concatenate((point_label1, point_label[org_len:],add_label_multi),axis=0)
                    point_inst = np.concatenate((point_inst1, point_inst[org_len:],add_inst_multi),axis=0)
                    # point_xyz.tofile('7.bin')
                    # point_label.tofile('7.label')
                    
                    if point_feat is not None:
                        add_fea =  points[:,3:]
                        add_fea_multi = points_multi[:,3:]
                        if len(add_fea.shape) == 2: add_fea = np.squeeze(add_fea)#[..., np.newaxis]
                        if len(add_fea_multi.shape) == 2: add_fea_multi = np.squeeze(add_fea_multi)#[..., np.newaxis]
                        
                        point_feat1 = np.concatenate((point_feat[:org_len],add_fea),axis=0)
                        point_feat =  np.concatenate((point_feat1, point_feat[org_len:],add_fea_multi),axis=0)
                    add_idx +=1
                    total_point_num += points.shape[0]
                    org_len = org_len1
                    if total_point_num>5000:
                        early_break=True
                        break
                # prevent adding too many points which cause GPU memory error
                if early_break: break
        org_len1 = org_len # 假如实例增强失败，则需要额外定义
        
        
        # instance mask
        mask = np.zeros_like(point_label,dtype=bool)
        for label in self.thing_list:
            mask[point_label == label] = True

        # create unqiue instance list
        inst_label = point_inst[mask].squeeze()
        unique_label = np.unique(inst_label)
        num_inst = len(unique_label)

        
        for inst in unique_label:
            # get instance index
            index = np.where(point_inst == inst)[0]
            # skip small instance
            if index.size<10: continue
            # get center
            center = np.mean(point_xyz[index,:],axis=0)

            if self.local_transformation:
                # random translation and rotation
                point_xyz[index,:] = self.local_tranform(point_xyz[index,:],center)
            
            # random flip instance based on it center 
            if self.random_flip:
                # get axis
                long_axis = [center[0], center[1]]/(center[0]**2+center[1]**2)**0.5
                short_axis = [-long_axis[1],long_axis[0]]
                # random flip
                flip_type = np.random.choice(5,1)
                if flip_type==3:
                    point_xyz[index,:2] = self.instance_flip(point_xyz[index,:2],[long_axis,short_axis],[center[0], center[1]],flip_type)
            
            # 20% random rotate
            random_num = np.random.random_sample()
            if self.random_rotate:
                if random_num>0.8 and inst & 0xFFFF > 0:
                    random_choice = np.random.random(20)*np.pi*2
                    fail_flag = True
                    for r in random_choice:
                        center_r = self.rotate_origin(center[np.newaxis,...],r)
                        # check if occluded
                        if self.check_occlusion(np.delete(point_xyz, index, axis=0),center_r[0],
                                                np.delete(point_label, index, axis=0)):
                            fail_flag = False
                            break
                    if not fail_flag:
                        # rotate to empty space
                        point_xyz[index,:] = self.rotate_origin(point_xyz[index,:],r)
        
        # instance augmentation for pole and traffic-sign
        # rotate the pole and traffic-sign in current instance
        # index_sign_pole = (point_label == 18) + (point_label == 19)
        # points_pole = point_xyz[index_sign_pole, :]
        # pole_label = point_label[index_sign_pole]
        # random_choice = np.random.random(2) * np.pi * 2
        # point_xyz1 = np.concatenate((point_xyz[:org_len1,:], self.rotate_origin(points_pole, random_choice[0]),
        #                             self.rotate_origin(points_pole, random_choice[1])), axis=0)
        # point_xyz = np.concatenate((point_xyz1, point_xyz[org_len1:,:]))
        # point_inst = np.concatenate((point_inst[:org_len1], point_inst[index_sign_pole], point_inst[index_sign_pole],point_inst[org_len1:]), axis=0)
        # if point_feat is not None:
        #     point_feat = np.concatenate((point_feat[:org_len1], point_feat[index_sign_pole], point_feat[index_sign_pole],point_feat[org_len1:]), axis=0)
        # point_label = np.concatenate((point_label[:org_len1], pole_label, pole_label, point_label[org_len1:]), axis=0)

        # add pole and traffic-sign to current instance from library
        pole_xyz_all = []
        pole_label_all = []
        pole_sig_all = []
        for inst_class in ['pole_and_traffic', 'pole', 'traffic']:
            instance_choice = np.random.choice(len(self.pole_instance_dict[inst_class]),self.add_num,replace=True)
            for instance_id in instance_choice:
                inst_points = np.fromfile(self.pole_instance_dict[inst_class][instance_id], dtype=np.float32).reshape(-1,4)
                inst_labels = np.fromfile(self.pole_instance_dict[inst_class][instance_id].replace('velodyne', 'label').replace('bin', 'label'),
                                            dtype=np.int32).reshape(-1,)
                xyz_pole = inst_points[:,:3]
                sig_pole = inst_points[:,3:]
                # sig_pole = np.ones((xyz_pole.shape[0],), dtype=np.float32)
                center_pole = np.mean(xyz_pole, axis=0)
                fail_flag = True
                # random rotate
                random_choice = np.random.random(20)*np.pi*2
                for r in random_choice:
                    center_r = self.rotate_origin(center_pole[np.newaxis,...],r)
                    # check if occluded
                    if self.check_occlusion(point_xyz,center_r[0], point_label):
                        fail_flag = False
                        break
                # rotate to empty space
                if fail_flag: continue
                xyz_pole = self.rotate_origin(xyz_pole, r)
                pole_xyz_all.append(xyz_pole)
                pole_label_all.append(inst_labels)
                pole_sig_all.append(sig_pole)
        if len(pole_xyz_all) > 0:
            point_xyz1 = np.concatenate((point_xyz[:org_len1,:], np.concatenate(pole_xyz_all, axis=0)), axis=0)
            point_xyz = np.concatenate((point_xyz1, point_xyz[org_len1:,:]))
            point_inst = np.concatenate((point_inst[:org_len1], np.concatenate(pole_label_all, axis=0),
                                        point_inst[org_len1:]), axis=0)
            if point_feat is not None:
                point_feat = np.concatenate((point_feat[:org_len1], np.squeeze(np.concatenate(pole_sig_all, axis=0)), 
                                            point_feat[org_len1:]), axis=0)
            point_label = np.concatenate((point_label[:org_len1], np.concatenate(pole_label_all, axis=0),
                                        point_label[org_len1:]), axis=0)
            point_label = point_label.astype(np.uint8)
            org_len1 = point_xyz1.shape[0]

        # point_xyz.tofile('8.bin')
        # point_label.tofile('8.label')
        if len(point_label.shape) == 1: point_label = point_label[..., np.newaxis]
        if len(point_inst.shape) == 1: point_inst = point_inst[..., np.newaxis]
        
        if point_feat is not None:
            return point_xyz,point_label,point_inst,point_feat, org_len1
        else:
            return point_xyz,point_label,point_inst, org_len1

    def instance_flip(self, points,axis,center,flip_type = 1):
        points = points[:]-center
        if flip_type == 1:
            # rotate 180 degree
            points = -points+center
        elif flip_type == 2:
            # flip over long axis
            a = axis[0][0]
            b = axis[0][1]
            flip_matrix = np.array([[b**2 - a**2, -2 * a * b],[-2 * a * b, a**2 - b**2]])
            points = np.matmul(flip_matrix,np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0))+center
        elif flip_type == 3:
            # flip over short axis
            a = axis[1][0]
            b = axis[1][1]
            flip_matrix = np.array([[b**2 - a**2, -2 * a * b],[-2 * a * b, a**2 - b**2]])
            points = np.matmul(flip_matrix,np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0))+center

        return points

    def check_occlusion(self,points,center,point_label,min_dist=2):
        'check if close to a point'
        points_road = points[point_label == 9, :]
        points_other = points[point_label != 9, :]
        if points.ndim == 1:
            dist = np.linalg.norm(points[np.newaxis,:]-center,axis=1)
        else:
            dist_other = np.linalg.norm(points_other-center,axis=1)
            dist_road = np.linalg.norm(points_road[:,:2]-center[:2],axis=1)
        return np.all(dist_other>min_dist) and np.any(dist_road<min_dist)

    def rotate_origin(self,xyz,radians):
        'rotate a point around the origin'
        x = xyz[:,0]
        y = xyz[:,1]
        new_xyz = xyz.copy()
        new_xyz[:,0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:,1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz

    def local_tranform(self,xyz,center):
        'translate and rotate point cloud according to its center'
        # random xyz
        loc_noise = np.random.normal(scale = 0.25, size=(1,3))
        # random angle
        rot_noise = np.random.uniform(-np.pi/20, np.pi/20)

        xyz = xyz-center
        xyz = self.rotate_origin(xyz,rot_noise)
        xyz = xyz+loc_noise
        
        return xyz+center

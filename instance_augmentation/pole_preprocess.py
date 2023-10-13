import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pc_dataset import SemKITTI_sk_multiscan

if __name__ == '__main__':
    # instance preprocessing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_path', default='/data1/SemanticKITTI/dataset')
    parser.add_argument('-o', '--out_path', default='/dev/JF/instance_aug/pole')

    args = parser.parse_args()

    train_pt_dataset = SemKITTI_sk_multiscan(data_path=args.data_path + '/sequences/', 
                                            imageset = 'train', return_ref = True,method='offset',
                                            label_mapping="instance_augmentation/semantic-kitti.yaml")
    train_pt_dataset.pole_preprocess(args.out_path)
    print('instance preprocessing finished.')
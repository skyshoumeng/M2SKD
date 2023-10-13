import pickle
import os


instance_pkl_path = "/dev/JF/instance_aug/registration/sequences/instance_path.pkl"
with open(instance_pkl_path, 'rb') as f:
    multi_instance_path = pickle.load(f)

class_name = ["car","bicycle","motorcycle","truck","other-vehicle","person","bicyclist","motorcyclist"]
print(multi_instance_path.keys())
for i, key in enumerate(multi_instance_path.keys()):
    print(class_name[i], len(multi_instance_path[key]))
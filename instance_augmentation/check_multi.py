import os
from tqdm import tqdm

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

cur_instance_root = r'/data1/SemanticKITTI/dataset/sequences/instance/sequences'
multi_root = r'/dev/JF/instance_aug/sequences'
split = [0,1,2,3,4,5,6,7,9,10]
im_idx = []
for i_folder in split:
    abs_path = absoluteFilePaths('/'.join([cur_instance_root, str(i_folder).zfill(2), 'instance']))
    im_idx += abs_path
multi_idx = []
for i_folder in split:
    abs_path = absoluteFilePaths('/'.join([multi_root, str(i_folder).zfill(2), 'instance']))
    multi_idx += abs_path
    
cur_idx = []
for c in im_idx:
    cur_idx.append(c.split('/sequences/',2)[2])
pbar = tqdm(total=len(multi_idx), ncols=100)
ss=0
for p in  multi_idx:
    p = p.split('/sequences/',1)[1]
    if not p in cur_idx:
        print(p)
        ss+=1
    pbar.update(1)
pbar.close()

# mul_idx = []
# for c in multi_idx:
#     mul_idx.append(c.split('/sequences/',1)[1])
# pbar = tqdm(total=len(im_idx), ncols=100)
# ss=0
# for p in  im_idx:
#     p = p.split('/sequences/',2)[2]
#     if not p in mul_idx:
#         print(p)
#         ss+=1
#     pbar.update(1)
# pbar.close()
if ss == 0:
    print("All multi-instance has corresponding single-instance")
else:
    print("There are {} frame(s) have not corresponding single-instance.".format(ss))
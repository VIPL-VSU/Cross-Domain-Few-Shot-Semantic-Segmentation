import numpy as np
import glob, tqdm, os
import pickle
from PIL import Image

scannet20 = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair','counter', 'curtain', 'desk', 'door', 'floor', 
             'otherfurniture', 'picture', 'refridgerator', 'shower curtain', 'sink', 'sofa', 'table', 'toilet', 'wall', 'window']
s3DIS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'column' , 'door', 'floor', 'sofa', 'table', 'wall', 'window']
NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

scannet20_name2id = {name.strip():i for i, name in enumerate(scannet20)}
# scannet_flag = [False for i in range(len(scannet20))]
scannet_count = [0 for i in range(len(scannet20))]
s3DIS_name2id = {name.strip():i for i, name in enumerate(s3DIS)}
# s3DIS_flag = [False for i in range(len(s3DIS))]
s3DIS_count = [0 for i in range(len(s3DIS))]


base_dir = '/home/xiaojiwei/workspace/3Ddet/HolisticPoseGraph/data/sunrgbd/sunrgbd_train_test_data'
data_files = glob.glob(os.path.join(base_dir, '*.pkl'))
for data_file in tqdm.tqdm(data_files):
    with open(data_file, 'rb') as t:
        img_data = pickle.load(t)

        image = Image.fromarray(img_data['rgb_img'])
        depth = img_data['depth_map']
        boxes = img_data['boxes']
        camera_k = img_data['camera']['K']
        bbox = boxes['bdb2D_pos'] #shape: n_object * 4
        cls_id = boxes['size_cls'] #shape: n_object * 1

        for idx in range(len(bbox)):
            classname = NYU40CLASSES[cls_id[idx]]
            bdb = bbox[idx]
            if classname in scannet20:
                img_save_dir = os.path.join('ScanNet', 'sunrgbd', 'images', classname)
                dep_save_dir = os.path.join('ScanNet', 'sunrgbd', 'depths', classname)
                cam_save_dir = os.path.join('ScanNet', 'sunrgbd', 'intrinsic', classname)
                os.makedirs(img_save_dir, exist_ok=True)
                os.makedirs(dep_save_dir, exist_ok=True)
                os.makedirs(cam_save_dir, exist_ok=True)
                id = scannet20_name2id[classname]
                num = scannet_count[id]
                img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
                img_save_name = os.path.join(img_save_dir, str(num)+'.jpg')
                img.save(img_save_name)
                dep = depth[bdb[1]:bdb[3], bdb[0]:bdb[2]]
                dep_save_name = os.path.join(dep_save_dir, str(num)+'.npy')
                cam_save_name = os.path.join(cam_save_dir, str(num)+'.npy')
                np.savetxt(dep_save_name, dep)
                np.savetxt(cam_save_name, camera_k)
                scannet_count[id] += 1
            elif classname in s3DIS:
                img_save_dir = os.path.join('S3DIS', 'sunrgbd', 'images', classname)
                dep_save_dir = os.path.join('S3DIS', 'sunrgbd', 'depths', classname)
                os.makedirs(img_save_dir, exist_ok=True)
                os.makedirs(dep_save_dir, exist_ok=True)
                id = s3DIS_name2id[classname]
                num = s3DIS_count[id]
                img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
                img_save_name = os.path.join(img_save_dir, str(num)+'.jpg')
                img.save(img_save_name)
                dep = depth[bdb[1]:bdb[3], bdb[0]:bdb[2]]
                dep_save_name = os.path.join(dep_save_dir, str(num)+'.npy')
                np.savetxt(dep_save_name, dep)
                s3DIS_count[id] += 1
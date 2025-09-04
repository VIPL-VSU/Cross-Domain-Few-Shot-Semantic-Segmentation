import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import os, glob, tqdm
import shutil

scannet20 = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair','counter', 'curtain', 'desk', 'door', 'floor', 
             'otherfurniture', 'picture', 'refridgerator', 'shower curtain', 'sink', 'sofa', 'table', 'toilet', 'wall', 'window']
s3DIS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'column' , 'door', 'floor', 'sofa', 'table', 'wall', 'window']

scannet20_name2id = {name.strip():i for i, name in enumerate(scannet20)}
scannet_flag = [False for i in range(len(scannet20))]
scannet_count = [0 for i in range(len(scannet20))]
s3DIS_name2id = {name.strip():i for i, name in enumerate(s3DIS)}
s3DIS_flag = [False for i in range(len(s3DIS))]
s3DIS_count = [0 for i in range(len(s3DIS))]
# 加载YOLOv5模型（使用PyTorch Hub）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 使用小模型 'yolov5s'

# 物体类别名称
labels = model.names  # 获取类别名称

def get_largest_object(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB格式以便绘图

    # 使用YOLO模型进行检测
    results = model(img_rgb)  # 传入图像

    # 提取检测结果（boxes, confidences, class_ids）
    boxes = results.xywh[0][:, :4].cpu().numpy()  # 获取边界框 (x, y, w, h)
    confidences = results.xywh[0][:, 4].cpu().numpy()  # 获取置信度
    class_ids = results.xywh[0][:, 5].cpu().numpy().astype(int)  # 获取类别ID

    if len(boxes) == 0:
        return None, None  # 没有检测到物体

    # 计算每个检测框的面积 (w * h)
    areas = boxes[:, 2] * boxes[:, 3]

    # 找到最大的物体
    max_idx = np.argmax(areas)

    # 获取最大的物体的类别
    largest_object_class = labels[class_ids[max_idx]]
    return largest_object_class, areas[max_idx]

# 示例使用
base_dir = '/home/xiaojiwei/data/scannet_images'
scene_list = glob.glob(os.path.join(base_dir, 'scene*'))
for scene in tqdm.tqdm(scene_list):
    print('Processing:', os.path.basename(scene))
    image_list = glob.glob(os.path.join(scene, 'color', '*.jpg'))
    for image_path in image_list:
        classname, area = get_largest_object(image_path)
        if classname in scannet20 and not all(scannet_flag):
            id = scannet20_name2id[classname]
            if scannet_flag[id]:
                continue
            num = scannet_count[id]
            save_dir = os.path.join('ScanNet', 'images', classname)
            os.makedirs(save_dir, exist_ok=True)
            save_name = os.path.join(save_dir, str(num)+'.jpg')
            shutil.copy(image_path, save_name)
            scannet_count[id] += 1
            if num == 100:
                scannet_flag[id] = True
            
        # if classname in s3DIS and not all(s3DIS_flag):
            



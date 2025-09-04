import os
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

def depth_to_point_cloud(depth_img, rgb_img, K):
    """
    将深度图和RGB图像转换为点云
    depth_img: numpy array, 深度图，尺寸为 (H, W)
    rgb_img: numpy array, RGB图像，尺寸为 (H, W, 3)
    K: numpy array, 相机内参矩阵，形状为 (3, 3)
    return: open3d.geometry.PointCloud, 转换后的点云
    """
    h, w = depth_img.shape
    points = []
    colors = []
    
    # 从内参矩阵中提取焦距和主点
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    for v in range(h):
        for u in range(w):
            # 深度值
            z = depth_img[v, u] # 假设深度值单位为米
            if z == 0:  # 跳过无效的深度值
                continue
            # 计算点云坐标
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            points.append([x, y, z])
            colors.append(rgb_img[v, u] / 255.0)  # 归一化RGB颜色值

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def process_sunrgbd_data(sunrgbd_root):
    """
    处理sunrgbd数据集，生成点云并保存为PLY文件
    sunrgbd_root: 数据集根目录，包含 'depths'、'images' 和 'intrinsic' 文件夹
    """
    depths_dir = os.path.join(sunrgbd_root, "depths")
    images_dir = os.path.join(sunrgbd_root, "images")
    intrinsic_dir = os.path.join(sunrgbd_root, "intrinsic")
    pointclouds_dir = os.path.join(sunrgbd_root, "pointclouds")
    
    # 创建输出目录
    if not os.path.exists(pointclouds_dir):
        os.mkdir(pointclouds_dir)
    
    # 遍历所有物体类别
    for category in os.listdir(depths_dir):
        category_depth_dir = os.path.join(depths_dir, category)
        category_image_dir = os.path.join(images_dir, category)
        category_intrinsic_dir = os.path.join(intrinsic_dir, category)
        category_pointcloud_dir = os.path.join(pointclouds_dir, category)
        
        # 创建物体类别子目录
        if not os.path.exists(category_pointcloud_dir):
            os.mkdir(category_pointcloud_dir)
        
        # 获取深度和图像的文件列表
        depth_files = os.listdir(category_depth_dir)
        
        # 处理每一对深度图、图像和相机内参矩阵
        for i in tqdm(range(len(depth_files))):
            # 读取深度图、RGB图像和相机内参矩阵
            depth_path = os.path.join(category_depth_dir, f'{i}.npy')
            image_path = os.path.join(category_image_dir,  f'{i}.jpg')
            intrinsic_path = os.path.join(category_intrinsic_dir,  f'{i}.npy')

            ply_filename = f"{i}.ply"
            ply_path = os.path.join(category_pointcloud_dir, ply_filename)

            if os.path.exists(ply_path):
                continue
            
            depth_img = np.loadtxt(depth_path)  # 深度图
            rgb_img = cv2.imread(image_path)  # RGB图像
            intrinsic_matrix = np.loadtxt(intrinsic_path)  # 相机内参矩阵

            if len(depth_img.shape) != 2:
                print("Skipping sample with invalid dimensions:", depth_img.shape)
                continue  # 跳过该样例

            # 生成点云
            pcd = depth_to_point_cloud(depth_img, rgb_img, intrinsic_matrix)
            
            # 保存点云为PLY文件
            o3d.io.write_point_cloud(ply_path, pcd)

if __name__ == "__main__":
    # 设置sunrgbd数据集的根目录路
    sunrgbd_root = '/home/xiaojiwei/workspace/3Dseg/PAP-FZS3D/datasets/ScanNet/sunrgbd'
    
    # 处理数据集
    process_sunrgbd_data(sunrgbd_root)

import os
import numpy as np
import open3d as o3d
import tqdm

def convert_ply_to_npy_with_color(root_path):
    # 定义点云文件夹路径
    pointclouds_path = os.path.join(root_path, "pointclouds")

    # 遍历类别文件夹
    for category in os.listdir(pointclouds_path):
        category_path = os.path.join(pointclouds_path, category)
        
        # 确保是一个目录
        if not os.path.isdir(category_path):
            continue
        
        # 遍历类别文件夹中的点云文件
        for file_name in tqdm.tqdm(os.listdir(category_path)):
            if file_name.endswith(".ply"):
                ply_file_path = os.path.join(category_path, file_name)
                npy_file_path = os.path.join(category_path, file_name.replace(".ply", ".npy"))
                
                # 读取 .ply 文件
                ply_data = o3d.io.read_point_cloud(ply_file_path)
                
                # 获取点云的坐标
                points = np.asarray(ply_data.points)
                # 获取点云的颜色
                colors = np.asarray(ply_data.colors)
                
                # 将坐标和颜色拼接
                if colors.size > 0:  # 检查颜色是否存在
                    data = np.hstack((points, colors))
                else:  # 如果没有颜色信息，仅保存点
                    data = points
                
                # 保存为 .npy 文件
                np.save(npy_file_path, data)
                print(f"Converted {ply_file_path} to {npy_file_path}")

# 主函数调用
if __name__ == "__main__":
    # 替换为你的 sunrgbd 根路径
    sunrgbd_root = "/home/xiaojiwei/workspace/3Dseg/PAP-FZS3D/datasets/ScanNet/sunrgbd"
    convert_ply_to_npy_with_color(sunrgbd_root)

import os
import shutil

def move_sam_files(root_path):
    # 定义原始点云文件夹和目标文件夹
    pointclouds_path = os.path.join(root_path, "pointclouds")
    target_path = os.path.join(root_path, "pointcloud_sam")
    
    # 确保目标文件夹存在
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 遍历类别子文件夹
    for category in os.listdir(pointclouds_path):
        category_path = os.path.join(pointclouds_path, category)
        
        # 确保是一个目录
        if not os.path.isdir(category_path):
            continue

        # 在目标文件夹中创建对应的类别子文件夹
        target_category_path = os.path.join(target_path, category)
        if not os.path.exists(target_category_path):
            os.makedirs(target_category_path)
        
        # 遍历子文件夹中的文件
        for file_name in os.listdir(category_path):
            if file_name.endswith("_sam.ply") or file_name.endswith("_sam.npy"):
                source_file_path = os.path.join(category_path, file_name)
                target_file_path = os.path.join(target_category_path, file_name)
                
                # 移动文件
                shutil.move(source_file_path, target_file_path)
                print(f"Moved {source_file_path} to {target_file_path}")

if __name__ == "__main__":
    # 替换为你的 sunrgbd 根路径
    sunrgbd_root = "/home/xiaojiwei/workspace/3Dseg/attMPTI/datasets/ScanNet/sunrgbd"
    move_sam_files(sunrgbd_root)

import numpy as np
from PIL import Image

# 载入npy文件
npy_file = 'ScanNet/sunrgbd/depths/sofa/15.npy'  # 替换为你的文件路径
depth_matrix = np.loadtxt(npy_file)

# 去掉多余的维度 (如果是 h * w * 1，可以通过 squeeze 压缩为 h * w)
depth_matrix = depth_matrix.squeeze()  # 结果应为 h * w

# 确保深度值为整数 (PNG一般以8位或16位保存)
# 如果深度范围较大，可能需要将其归一化到 [0, 255] 范围，或 [0, 65535] 范围
# 这里以 [0, 255] 归一化为例

# 归一化到 0-255 范围
depth_min = np.min(depth_matrix)
depth_max = np.max(depth_matrix)
depth_normalized = 255 * (depth_matrix - depth_min) / (depth_max - depth_min)

# 转换为 uint8 类型
depth_normalized = depth_normalized.astype(np.uint8)

# 将深度数据保存为 PNG 文件
depth_image = Image.fromarray(depth_normalized)
depth_image.save('ScanNet/show_depth/depth_15.png')

print("Depth image saved as 'depth_image.png'")

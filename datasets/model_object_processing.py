import glob, tqdm, os
import numpy as np

import random

scannet20 = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair','counter', 'curtain', 'desk', 'door', 'floor', 
             'otherfurniture', 'picture', 'refridgerator', 'shower curtain', 'sink', 'sofa', 'table', 'toilet', 'wall', 'window']
s3DIS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'column' , 'door', 'floor', 'sofa', 'table', 'wall', 'window']

ModelNet40_DIR = '/home/xiaojiwei/data/ModelNet40'
ModelNet40_Files = glob.glob(os.path.join(ModelNet40_DIR, '*'))
ModelNet40_CLASSNAMES = [os.path.basename(x) for x in ModelNet40_Files]

def read_off_and_sample_points(off_file_path, npy_file_path, num_samples_per_face=10):
    """
    Reads an OFF file containing point cloud data and samples points uniformly on the faces.

    Args:
        off_file_path (str): Path to the input OFF file.
        npy_file_path (str): Path to the output .npy file.
        num_samples_per_face (int): Number of points to sample per face.
    """
    def sample_point_on_triangle(v0, v1, v2):
        """Uniformly sample a point on a triangle defined by vertices v0, v1, v2."""
        u = random.random()
        v = random.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        w = 1 - u - v
        return u * v0 + v * v1 + w * v2

    with open(off_file_path, 'r') as f:
        # Read the first line
        header = f.readline().strip()
        
        # Handle cases where the first line contains both "OFF" and metadata
        if header.startswith("OFF"):
            parts = header[3:].strip().split()  # Extract data after "OFF"
            if parts:  # If there are additional parts, treat them as metadata
                n_vertices, n_faces, _ = map(int, parts)
            else:  # If only "OFF" is present, read the next line for metadata
                n_vertices, n_faces, _ = map(int, f.readline().strip().split())
        else:
            raise ValueError(f"Unsupported file format: {header}. Expected 'OFF'.")

        # Read the vertex data (x, y, z)
        vertices = []
        for _ in range(n_vertices):
            line = f.readline().strip().split()
            x, y, z = map(float, line[:3])
            vertices.append([x, y, z])
        vertices = np.array(vertices, dtype=np.float32)

        # Read the face data
        faces = []
        for _ in range(n_faces):
            line = f.readline().strip().split()
            face_indices = list(map(int, line[1:]))  # Skip the first number (number of vertices per face)
            faces.append(face_indices)

        # Sample points on the faces
        sampled_points = []
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            for _ in range(num_samples_per_face):
                sampled_points.append(sample_point_on_triangle(v0, v1, v2))

        sampled_points = np.array(sampled_points, dtype=np.float32)
        colors = np.ones((len(sampled_points), 3)) * 255

        # 将坐标矩阵和颜色矩阵合并为 n x 6 的矩阵
        sampled_points = np.hstack((sampled_points, colors))

        if len(sampled_points) < 2048:
            return
        # Save sampled points as .npy file
        np.save(npy_file_path, sampled_points)
        print(f"Saved {len(sampled_points)} sampled points to {npy_file_path}")


def process_modelnet40_object(basedir, classname, dataset, split='train'):
    data_dir = os.path.join(basedir, classname, split)
    save_dir = os.path.join(dataset, 'modelnet_objects', classname, split)
    os.makedirs(save_dir, exist_ok=True)
    object_files = glob.glob(os.path.join(data_dir, '*.off'))
    for object_file in object_files:
        object_name = os.path.basename(object_file)
        save_file = os.path.join(save_dir, object_name.replace('.off','.npy'))
        read_off_and_sample_points(object_file, save_file)


if __name__ == "__main__":
    for modelnet_classname in ModelNet40_CLASSNAMES:
        if modelnet_classname in scannet20:
            process_modelnet40_object(ModelNet40_DIR, modelnet_classname, 'ScanNet', 'train')
            process_modelnet40_object(ModelNet40_DIR, modelnet_classname, 'ScanNet', 'test')
        if modelnet_classname in s3DIS:
            process_modelnet40_object(ModelNet40_DIR, modelnet_classname, 'S3DIS', 'train')
            process_modelnet40_object(ModelNet40_DIR, modelnet_classname, 'S3DIS', 'test')
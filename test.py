import trimesh
import numpy as np

# 加载网格文件
mesh = trimesh.load('cam.stl')

# 指定采样点数
num_points = 2024

# 从网格采样点云
sampled_points = mesh.sample(num_points)

# 获取点云数据（包括坐标和法线）
point_cloud = sampled_points
print(point_cloud)
# 输出点云数据的形状
print(f"Point cloud shape: {point_cloud.shape}")  # 输出: (num_points, 3)

# 创建点云对象
point_cloud_obj = trimesh.points.PointCloud(point_cloud)

# 可视化点云
point_cloud_obj.show()
import open3d as o3d
import numpy as np
import os
import cv2

# 从点云文件加载点云数据
def load_point_cloud(frame_id, load_dir="point_cloud_frames"):
    file_path = os.path.join(load_dir, f"frame_{frame_id:04d}.ply")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    point_cloud = o3d.io.read_point_cloud(file_path)
    return point_cloud

# 渲染点云并返回图像
def render_point_cloud(point_cloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 不显示窗口
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    # 捕获渲染图像
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)  # 转换为8位图像
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换颜色空间

    vis.destroy_window()
    return image

# 渲染所有帧并保存为视频
def render_and_save_video(num_frames=100, load_dir="point_cloud_frames", output_video_path="output_video.mp4", fps=30):
    # 获取第一帧的分辨率
    first_frame = load_point_cloud(0, load_dir)
    first_image = render_point_cloud(first_frame)
    height, width, _ = first_image.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_id in range(num_frames):
        # 加载点云数据
        point_cloud = load_point_cloud(frame_id, load_dir)

        # 渲染点云为图像
        image = render_point_cloud(point_cloud)

        # 写入视频帧
        video_writer.write(image)
        print(f"渲染帧 {frame_id} 并写入视频")

    # 释放视频写入器
    video_writer.release()
    print(f"视频已保存到: {output_video_path}")

# 运行脚本
render_and_save_video(num_frames=100, load_dir="point_cloud_frames", output_video_path="output_video.mp4", fps=30)
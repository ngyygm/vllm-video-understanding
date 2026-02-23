"""
视频截图工具
将视频按照设定的每秒帧数截图保存为图片
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path


def video_to_frames(video_path: str, output_dir: str, fps: float = 1.0):
    """
    将视频转换为图片帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出图片目录
        fps: 每秒截图数量（默认1.0，即每秒1张）
    """
    # 转换为绝对路径
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 创建输出目录结构
    video_name = Path(video_path).stem
    output_folder = os.path.join(output_dir, video_name, "frames")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"输出文件夹绝对路径: {output_folder}")
    print(f"文件夹是否存在: {os.path.exists(output_folder)}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)  # 视频原始帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    duration = total_frames / video_fps  # 视频时长（秒）
    
    print(f"视频信息:")
    print(f"  - 原始帧率: {video_fps:.2f} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {duration:.2f} 秒")
    print(f"  - 设定截图频率: {fps} 张/秒")
    print(f"  - 预计输出图片数: {int(duration * fps)} 张")
    print(f"  - 输出目录: {output_folder}")
    print("-" * 50)
    
    # 计算采样间隔（每隔多少帧截取一张）
    frame_interval = video_fps / fps
    
    frame_count = 0
    saved_count = 0
    next_frame_to_save = 0
    errors = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 判断是否需要保存当前帧
        if frame_count >= next_frame_to_save:
            # 生成文件名（使用时间戳命名，格式如 frame_001.23s.jpg）
            timestamp = frame_count / video_fps
            filename = f"frame_{timestamp:07.2f}s.jpg"
            output_path = os.path.join(output_folder, filename)
            
            # 保存图片（使用 imencode 解决中文路径问题）
            try:
                success, encoded_img = cv2.imencode('.jpg', frame)
                if success:
                    with open(output_path, 'wb') as f:
                        f.write(encoded_img.tobytes())
                    saved_count += 1
                else:
                    errors.append(f"编码失败: {output_path}")
            except Exception as e:
                errors.append(f"保存失败: {output_path}, 错误: {str(e)}")
            
            next_frame_to_save += frame_interval
            
            # 显示进度
            if saved_count % 10 == 0 or saved_count <= 5:
                print(f"已保存 {saved_count} 张图片 (时间: {timestamp:.2f}s)")
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    
    print("-" * 50)
    
    if errors:
        print(f"警告: 有 {len(errors)} 个文件保存失败")
        for err in errors[:5]:  # 只显示前5个错误
            print(f"  - {err}")
    
    print(f"完成！共保存 {saved_count} 张图片到: {output_folder}")
    
    # 检查文件夹内容
    files = os.listdir(output_folder) if os.path.exists(output_folder) else []
    print(f"文件夹内文件数: {len(files)}")
    
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="视频截图工具 - 将视频按照设定的频率转换为图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python video_to_frames.py --video data/摆放剃须刀.mp4 --fps 3
  python video_to_frames.py --video data/摆放剃须刀.mp4 --fps 0.5 --output ./frames
  python video_to_frames.py -v data/摆放剃须刀.mp4 -f 10 -o ./output
        """
    )
    
    parser.add_argument(
        '-v', '--video',
        type=str,
        required=True,
        help='输入视频文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output',
        help='输出图片目录 (默认: ./output)'
    )
    
    parser.add_argument(
        '-f', '--fps',
        type=float,
        default=1.0,
        help='每秒截图数量 (默认: 1.0)'
    )
    
    args = parser.parse_args()
    
    # 验证fps参数
    if args.fps <= 0:
        raise ValueError("fps 必须大于 0")
    
    # 执行转换
    video_to_frames(args.video, args.output, args.fps)


if __name__ == "__main__":
    main()

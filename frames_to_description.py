"""
图片组描述工具
将图片按组发送给 LLM 进行视频内容描述
"""

import os
import base64
import json
import argparse
from pathlib import Path
from openai import OpenAI
import glob


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_files(frames_dir: str) -> list:
    """获取目录下所有图片文件（按文件名排序）"""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(frames_dir, ext)))
    
    # 按文件名排序
    image_files.sort()
    return image_files


def group_images(image_files: list, group_size: int = 10, overlap: int = 0) -> list:
    """
    将图片分组，支持重叠
    
    Args:
        image_files: 图片文件列表
        group_size: 每组图片数量
        overlap: 组之间的重叠帧数
    
    Returns:
        分组后的图片列表
    """
    if overlap >= group_size:
        raise ValueError("overlap 必须小于 group_size")
    
    groups = []
    step = group_size - overlap  # 步长
    i = 0
    
    while i < len(image_files):
        group = image_files[i:i + group_size]
        groups.append(group)
        i += step
        
        # 如果最后一组已经包含到最后一张图片，结束
        if i + group_size > len(image_files) and i >= len(image_files):
            break
    
    # 确保最后一张图片被包含
    if groups and len(groups[-1]) < group_size:
        # 最后一个分组不足，检查是否需要补充
        last_image = image_files[-1]
        if last_image not in groups[-1]:
            # 添加最后一个分组
            last_group = image_files[-(group_size):]
            if last_group not in groups:
                groups.append(last_group)
    
    return groups


def describe_frame_group(
    image_paths: list,
    client: OpenAI,
    model: str,
    system_prompt: str,
    group_index: int
) -> str:
    """使用 LLM 描述一组图片"""
    
    # 构建消息内容
    content = []
    
    # 添加所有图片
    for img_path in image_paths:
        base64_image = encode_image_to_base64(img_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # 调用 API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        max_tokens=2000
    )
    
    return response.choices[0].message.content


def save_result(output_file: str, result: dict):
    """保存结果到JSON文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def process_video_frames(
    frames_dir: str,
    group_size: int = 10,
    overlap: int = 0,
    llm_api_key: str = "ollama",
    llm_model: str = "qwen3:8b",
    llm_base_url: str = "http://127.0.0.1:11434/v1"
):
    """
    处理视频帧并生成描述
    
    Args:
        frames_dir: 图片目录
        group_size: 每组图片数量
        overlap: 组之间的重叠帧数
        llm_api_key: LLM API密钥
        llm_model: LLM模型名称
        llm_base_url: LLM API基础URL
    """
    
    # 系统提示词
    system_prompt = """你是一个专业的视频内容解析系统，你需要把视频内容详尽得用文字形式如实描述出来。你的目标是让一个完全看不到视频的人，仅通过你的描述，在脑海中完整重建这个视频。"""
    
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=llm_api_key,
        base_url=llm_base_url
    )
    
    # 获取图片文件
    print(f"正在读取图片目录: {frames_dir}")
    frames_dir = os.path.abspath(frames_dir)
    image_files = get_image_files(frames_dir)
    
    if not image_files:
        print("错误: 未找到图片文件")
        return
    
    print(f"共找到 {len(image_files)} 张图片")
    
    # 分组
    groups = group_images(image_files, group_size, overlap)
    print(f"分为 {len(groups)} 组 (每组 {group_size} 张，重叠 {overlap} 帧)")
    print("-" * 50)
    
    print(f"共 {len(groups)} 组，每组的描述将保存到单独的JSON文件")
    print("-" * 50)
    
    # 处理每一组
    for i, group in enumerate(groups):
        print(f"\n正在处理第 {i + 1}/{len(groups)} 组...")
        group_images_names = [os.path.basename(p) for p in group]
        print(f"  图片: {group_images_names}")
        
        try:
            description = describe_frame_group(
                image_paths=group,
                client=client,
                model=llm_model,
                system_prompt=system_prompt,
                group_index=i
            )
            
            print(f"  描述完成: {description[:100]}...")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            description = f"处理失败: {str(e)}"
        
        # 保存到单独的JSON文件
        group_result = {
            "images": group_images_names,
            "description": description
        }
        
        output_file = os.path.join(frames_dir, f"group_{i + 1}.json")
        save_result(output_file, group_result)
        print(f"  已保存到: {output_file}")
    
    print("\n" + "=" * 50)
    print(f"全部完成！共保存 {len(groups)} 个JSON文件到: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="图片组描述工具 - 使用 LLM 描述视频内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python frames_to_description.py --frames output/摆放剃须刀
  python frames_to_description.py --frames output/摆放剃须刀 --group-size 10 --overlap 2
  python frames_to_description.py -f output/摆放剃须刀 -g 9 -p 3 --model gemma3:27b
        """
    )
    
    parser.add_argument(
        '-f', '--frames',
        type=str,
        required=True,
        help='图片目录路径'
    )
    
    parser.add_argument(
        '-g', '--group-size',
        type=int,
        default=10,
        help='每组图片数量 (默认: 10)'
    )
    
    parser.add_argument(
        '-p', '--overlap',
        type=int,
        default=0,
        help='组之间的重叠帧数 (默认: 0)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default='ollama',
        help='LLM API密钥 (默认: ollama)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemma3:27b',
        help='LLM模型名称 (默认: qwen3:8b)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default='http://127.0.0.1:11434/v1',
        help='LLM API基础URL (默认: http://127.0.0.1:11434/v1)'
    )
    
    args = parser.parse_args()
    
    process_video_frames(
        frames_dir=args.frames,
        group_size=args.group_size,
        overlap=args.overlap,
        llm_api_key=args.api_key,
        llm_model=args.model,
        llm_base_url=args.base_url
    )


if __name__ == "__main__":
    main()

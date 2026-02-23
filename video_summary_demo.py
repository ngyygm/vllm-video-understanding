"""
视频内容总结工具
完整流程: 视频 -> 图片帧 -> 分组描述 -> 分段总结
支持按时间窗口（如每10秒）生成多个总结文件
"""

import os
import json
import argparse
import glob
import re
from pathlib import Path
from openai import OpenAI

# 导入现有模块
from image_processing.video_to_frames import video_to_frames
from image_processing.frames_to_description import (
    get_image_files,
    group_images,
    describe_frame_group,
    save_result
)


def extract_timestamp_from_filename(filename: str) -> float:
    """
    从文件名中提取时间戳（秒）
    例如: frame_0001.23s.jpg -> 1.23
    """
    match = re.search(r'frame_(\d+\.?\d*)s\.jpg', filename)
    if match:
        return float(match.group(1))
    return 0.0


def collect_json_descriptions(group_dir: str) -> list:
    """
    收集目录下所有 group_*.json 文件的描述内容
    
    Args:
        group_dir: group目录路径
    
    Returns:
        按顺序排列的描述列表
    """
    import re
    json_files = glob.glob(os.path.join(group_dir, "group_*.json"))
    # 按数字顺序排序，而不是字符串排序
    json_files.sort(key=lambda x: int(re.search(r'group_(\d+)\.json', os.path.basename(x)).group(1)))
    descriptions = []
    
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            descriptions.append({
                "file": os.path.basename(json_file),
                "images": data.get("images", []),
                "description": data.get("description", "")
            })
    
    return descriptions


def group_descriptions_by_count(
    descriptions: list,
    group_count: int
) -> list:
    """
    按group数量将描述分组
    
    Args:
        descriptions: 描述列表
        group_count: 每组包含的group数量
    
    Returns:
        按group数量分组的描述列表
    """
    if group_count <= 0:
        # 如果group_count为0或负数，返回所有描述作为一个组
        return [descriptions]
    
    windows = []
    for i in range(0, len(descriptions), group_count):
        window = descriptions[i:i + group_count]
        windows.append(window)
    
    return windows


def group_descriptions_by_time_window(
    descriptions: list,
    time_window: float,
    frames_dir: str
) -> list:
    """
    按时间窗口将描述分组
    
    Args:
        descriptions: 描述列表
        time_window: 时间窗口大小（秒）
        frames_dir: 帧图片目录，用于获取时间戳
    
    Returns:
        按时间窗口分组的描述列表
    """
    if time_window <= 0:
        # 如果时间窗口为0或负数，返回所有描述作为一个组
        return [descriptions]
    
    windows = []
    current_window = []
    current_window_end = time_window
    
    for desc in descriptions:
        # 从第一张图片文件名中提取时间戳
        if desc['images']:
            first_image = desc['images'][0]
            timestamp = extract_timestamp_from_filename(first_image)
            
            if timestamp < current_window_end:
                current_window.append(desc)
            else:
                # 当前窗口已满，开始新窗口
                if current_window:
                    windows.append(current_window)
                current_window = [desc]
                current_window_end = timestamp + time_window
    
    # 添加最后一个窗口
    if current_window:
        windows.append(current_window)
    
    return windows


def summarize_descriptions(
    descriptions: list,
    client: OpenAI,
    model: str,
    output_file: str = None,
    time_range: tuple = None
) -> str:
    """
    使用LLM总结所有分组描述
    
    Args:
        descriptions: 描述列表
        client: OpenAI客户端
        model: 模型名称
        output_file: 输出文件路径（可选）
        time_range: 时间范围 (start_time, end_time)（可选）
    
    Returns:
        总结内容
    """
    system_prompt = """你是一个专业的视频内容分析专家。现在有若干段视频片段的文字描述，这些描述是按照时间顺序排列的。
请将这些分段描述整合成一份完整、流畅的视频内容描述报告。

要求：
1. 按时间顺序组织内容
2. 去除重复和冗余的信息
3. 保持描述的连贯性和逻辑性
4. 突出视频中的关键动作、场景变化和重要细节
5. 使用清晰、准确的语言

直接输出整合后的完整描述内容即可，不需要任何标题或分段标记。
"""
    
    # 构建分段描述文本
    segments_text = []
    for i, desc in enumerate(descriptions, 1):
        segment = f"\n--- 第 {i} 段 (图片: {', '.join(desc['images'][:3])}... 共{len(desc['images'])}张) ---\n"
        segment += desc['description']
        segments_text.append(segment)
    
    full_text = "\n".join(segments_text)
    
    time_info = ""
    if time_range:
        time_info = f" (时间范围: {time_range[0]:.2f}s - {time_range[1]:.2f}s)"
    
    print(f"\n正在生成总结{time_info}...")
    print(f"  共 {len(descriptions)} 个分段描述")
    print(f"  总字符数: {len(full_text)}")
    
    # 调用LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是视频各片段的描述，请整合成一份完整的视频内容报告：\n{full_text}"}
        ],
        max_tokens=4000
    )
    
    summary = response.choices[0].message.content
    
    # 保存总结
    if output_file:
        result = {
            "time_range": time_range,
            "total_segments": len(descriptions),
            "summary": summary,
            "segments": descriptions
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  总结已保存到: {output_file}")
    
    return summary


def process_video(
    video_path: str,
    output_dir: str = "./output",
    fps: float = 3.0,
    group_size: int = 10,
    overlap: int = 2,
    time_window: float = 10.0,
    summary_group_count: int = 0,
    llm_api_key: str = "ollama",
    llm_model: str = "qwen3-vl:32b",
    llm_base_url: str = "http://127.0.0.1:11434/v1",
    summary_model: str = None,
    summary_api_key: str = None,
    summary_base_url: str = None,
    skip_frames: bool = False,
    skip_descriptions: bool = False
):
    """
    完整的视频处理流程
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        fps: 每秒截图数量
        group_size: 每组图片数量
        overlap: 组之间的重叠帧数
        summary_group_count: 按group数量分组，每组包含的group数量，0表示生成单个完整总结
        llm_api_key: 图像理解LLM API密钥
        llm_model: 图像理解LLM模型名称
        llm_base_url: 图像理解LLM API基础URL
        summary_model: 总结LLM模型名称（如果未设置，使用llm_model）
        summary_api_key: 总结LLM API密钥（如果未设置，使用llm_api_key）
        summary_base_url: 总结LLM API基础URL（如果未设置，使用llm_base_url）
        skip_frames: 是否跳过视频转帧步骤（如果已有帧图片）
        skip_descriptions: 是否跳过描述生成步骤（如果已有描述JSON）
    """
    print("=" * 60)
    print("视频内容总结工具")
    print("=" * 60)
    
    # 转换为绝对路径
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    
    # 获取视频名称
    video_name = Path(video_path).stem
    base_output_dir = os.path.join(output_dir, video_name)
    
    # 创建三个子文件夹
    frames_dir = os.path.join(base_output_dir, "frames")
    group_dir = os.path.join(base_output_dir, "group")
    summary_dir = os.path.join(base_output_dir, "summary")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(group_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    print(f"\n视频文件: {video_path}")
    print(f"输出目录: {base_output_dir}")
    print(f"  - 帧图片: {frames_dir}")
    print(f"  - 分组描述: {group_dir}")
    print(f"  - 总结: {summary_dir}")
    print(f"截图频率: {fps} 张/秒")
    print(f"分组设置: 每组 {group_size} 张, 重叠 {overlap} 帧")
    if summary_group_count > 0:
        print(f"总结分组: 每 {summary_group_count} 个group生成一个总结")
    else:
        print(f"总结方式: 生成单个完整总结")
    
    # 确定总结模型配置（如果未设置，使用图像理解模型配置）
    final_summary_model = summary_model if summary_model else llm_model
    final_summary_api_key = summary_api_key if summary_api_key else llm_api_key
    final_summary_base_url = summary_base_url if summary_base_url else llm_base_url
    
    print(f"图像理解模型: {llm_model}")
    print(f"总结模型: {final_summary_model}")
    if summary_model or summary_api_key or summary_base_url:
        print(f"  (使用独立总结模型配置)")
    print("-" * 60)
    
    # 步骤1: 视频转帧
    if not skip_frames:
        print("\n【步骤 1/3】 视频转帧...")
        # 临时修改输出目录，因为video_to_frames会在output_dir下创建video_name/frames
        video_to_frames(video_path, output_dir, fps)
    else:
        print(f"\n【步骤 1/3】 跳过视频转帧（使用已有帧）")
    
    # 步骤2: 生成分组描述
    if not skip_descriptions:
        print("\n【步骤 2/3】 生成分组描述...")
        
        # 初始化 OpenAI 客户端
        client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url
        )
        
        # 系统提示词
        system_prompt = """你是一个专业的视频内容解析系统，你需要把视频内容详尽得用文字形式如实描述出来。你的目标是让一个完全看不到视频的人，仅通过你的描述，在脑海中完整重建这个视频。"""
        
        # 获取图片文件
        image_files = get_image_files(frames_dir)
        
        if not image_files:
            print("错误: 未找到图片文件")
            return
        
        print(f"共找到 {len(image_files)} 张图片")
        
        # 分组
        groups = group_images(image_files, group_size, overlap)
        print(f"分为 {len(groups)} 组 (每组 {group_size} 张，重叠 {overlap} 帧)")
        print("-" * 50)
        
        # 处理每一组
        for i, group in enumerate(groups):
            print(f"\n正在处理第 {i + 1}/{len(groups)} 组...")
            group_images_names = [os.path.basename(p) for p in group]
            print(f"  图片: {group_images_names[:3]}... (共{len(group)}张)")
            
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
            
            # 保存到group文件夹
            group_result = {
                "images": group_images_names,
                "description": description
            }
            
            output_file = os.path.join(group_dir, f"group_{i + 1}.json")
            save_result(output_file, group_result)
            print(f"  已保存到: {output_file}")
    else:
        print(f"\n【步骤 2/3】 跳过描述生成（使用已有描述）")
    
    # 步骤3: 生成总结
    print("\n【步骤 3/3】 生成总结...")
    
    # 初始化总结模型客户端（使用独立配置或图像理解模型配置）
    summary_client = OpenAI(
        api_key=final_summary_api_key,
        base_url=final_summary_base_url
    )
    
    # 收集所有描述
    descriptions = collect_json_descriptions(group_dir)
    
    if not descriptions:
        print("错误: 未找到描述文件")
        return
    
    # 按group数量分组
    if summary_group_count > 0:
        # 优先使用按group数量分组
        print(f"\n按group数量 ({summary_group_count}个/组) 分组描述...")
        summary_windows = group_descriptions_by_count(
            descriptions, summary_group_count
        )
        print(f"分为 {len(summary_windows)} 个总结组")
        
        # 为每个总结组生成总结
        for i, window_descriptions in enumerate(summary_windows):
            # 计算时间范围
            if window_descriptions:
                first_image = window_descriptions[0]['images'][0] if window_descriptions[0]['images'] else ""
                last_image = window_descriptions[-1]['images'][-1] if window_descriptions[-1]['images'] else ""
                start_time = extract_timestamp_from_filename(first_image)
                end_time = extract_timestamp_from_filename(last_image)
                time_range = (start_time, end_time)
            else:
                time_range = None
            
            summary_file = os.path.join(summary_dir, f"summary_{i + 1:03d}_{start_time:.1f}s-{end_time:.1f}s.json")
            summary = summarize_descriptions(
                descriptions=window_descriptions,
                client=summary_client,
                model=final_summary_model,
                output_file=summary_file,
                time_range=time_range
            )
            
            print(f"\n总结组 {i + 1}/{len(summary_windows)} 完成")
            print(f"  包含 {len(window_descriptions)} 个group")
            print(f"  时间范围: {start_time:.2f}s - {end_time:.2f}s")
            print(f"  总结预览: {summary[:100]}...")
    else:
        # 生成单个完整总结
        print(f"\n生成完整总结...")
        summary_file = os.path.join(summary_dir, "summary_full.json")
        summary = summarize_descriptions(
            descriptions=descriptions,
            client=summary_client,
            model=final_summary_model,
            output_file=summary_file
        )
        
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"\n总结报告:")
        print("-" * 60)
        print(summary)
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"\n文件输出位置: {base_output_dir}")
    print(f"  - 图片帧: {frames_dir}/frame_*.jpg")
    print(f"  - 分组描述: {group_dir}/group_*.json")
    if summary_group_count > 0:
        print(f"  - 分段总结: {summary_dir}/summary_*.json")
    else:
        print(f"  - 完整总结: {summary_dir}/summary_full.json")


def main():
    parser = argparse.ArgumentParser(
        description="视频内容总结工具 - 完整流程处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整处理流程（每5个group一个总结，默认）
  python video_summary_demo.py --video data/摆放剃须刀.mp4
  
  # 每10个group生成一个总结
  python video_summary_demo.py --video data/摆放剃须刀.mp4 --summary-group-count 10
  
  # 生成单个完整总结
  python video_summary_demo.py --video data/摆放剃须刀.mp4 --summary-group-count 0
  
  # 自定义参数
  python video_summary_demo.py --video data/摆放剃须刀.mp4 --fps 5 --group-size 8 --overlap 2 --summary-group-count 5
  
  # 使用不同的模型
  python video_summary_demo.py -v data/摆放剃须刀.mp4 --model qwen3:8b --time-window 10
  
  # 使用独立的总结模型（图像理解用视觉模型，总结用文本模型）
  python video_summary_demo.py -v data/摆放剃须刀.mp4 --model gemma3:27b --summary-model gpt-4 --summary-base-url https://api.openai.com/v1
  
  # 如果已有帧图片，跳过视频转帧
  python video_summary_demo.py --video data/摆放剃须刀.mp4 --skip-frames --time-window 10
  
  # 如果已有描述JSON，跳过描述生成
  python video_summary_demo.py --video data/摆放剃须刀.mp4 --skip-descriptions --time-window 10
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
        help='输出目录 (默认: ./output)'
    )
    
    parser.add_argument(
        '-f', '--fps',
        type=float,
        default=3.0,
        help='每秒截图数量 (默认: 3.0)'
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
        default=2,
        help='组之间的重叠帧数 (默认: 2)'
    )
    
    parser.add_argument(
        '-c', '--summary-group-count',
        type=int,
        default=5,
        help='按group数量分组，每N个group生成一个总结。设为0表示生成单个完整总结 (默认: 5)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default='ollama',
        help='图像理解LLM API密钥 (默认: ollama)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemma3:27b',
        help='图像理解LLM模型名称 (默认: gemma3:27b)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default='http://127.0.0.1:11434/v1',
        help='图像理解LLM API基础URL (默认: http://127.0.0.1:11434/v1)'
    )
    
    parser.add_argument(
        '--summary-model',
        type=str,
        default=None,
        help='总结LLM模型名称（如果未设置，使用--model指定的模型）'
    )
    
    parser.add_argument(
        '--summary-api-key',
        type=str,
        default=None,
        help='总结LLM API密钥（如果未设置，使用--api-key指定的密钥）'
    )
    
    parser.add_argument(
        '--summary-base-url',
        type=str,
        default=None,
        help='总结LLM API基础URL（如果未设置，使用--base-url指定的URL）'
    )
    
    parser.add_argument(
        '--skip-frames',
        action='store_true',
        help='跳过视频转帧步骤（使用已有帧）'
    )
    
    parser.add_argument(
        '--skip-descriptions',
        action='store_true',
        help='跳过描述生成步骤（使用已有描述）'
    )
    
    args = parser.parse_args()
    
    # 验证fps参数
    if args.fps <= 0:
        raise ValueError("fps 必须大于 0")
    
    # 执行处理
    process_video(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        group_size=args.group_size,
        overlap=args.overlap,
        time_window=0,  # 已移除时间窗口方式
        summary_group_count=args.summary_group_count,
        llm_api_key=args.api_key,
        llm_model=args.model,
        llm_base_url=args.base_url,
        summary_model=args.summary_model,
        summary_api_key=args.summary_api_key,
        summary_base_url=args.summary_base_url,
        skip_frames=args.skip_frames,
        skip_descriptions=args.skip_descriptions
    )


if __name__ == "__main__":
    main()

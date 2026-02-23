# LLM Video Understanding

一个基于多模态大语言模型的长视频内容理解工具。通过将视频转换为帧图片，使用LLM进行内容描述，并生成结构化的文本总结。特别适用于长视频的深度理解和分析。

## ✨ 功能特性

- 🎬 **视频转帧** - 将视频按指定频率转换为带时间戳的帧图片
- 🖼️ **分组描述** - 将帧图片分组，使用多模态LLM生成详细描述
- 📝 **智能总结** - 整合所有描述，生成完整流畅的视频内容总结
- ⏱️ **时间窗口** - 支持按时间窗口（如每10秒）生成多个分段总结
- 📁 **结构化输出** - 自动组织输出文件到 frames、group、summary 三个文件夹

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```bash
# 完整处理流程（每10秒生成一个总结）
python video_summary_demo.py --video data/your_video.mp4 --time-window 10

# 生成单个完整总结
python video_summary_demo.py --video data/your_video.mp4 --time-window 0
```

## 📖 详细用法

### 完整流程处理

```bash
python video_summary_demo.py \
    --video data/your_video.mp4 \
    --fps 3.0 \
    --group-size 10 \
    --overlap 2 \
    --time-window 10.0 \
    --model gemma3:27b
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-v, --video` | 必需 | 输入视频文件路径 |
| `-o, --output` | `./output` | 输出目录 |
| `-f, --fps` | `3.0` | 每秒截图数量 |
| `-g, --group-size` | `10` | 每组图片数量 |
| `-p, --overlap` | `2` | 组之间的重叠帧数 |
| `-t, --time-window` | `10.0` | 时间窗口大小（秒），0表示生成单个完整总结 |
| `--model` | `gemma3:27b` | LLM模型名称 |
| `--base-url` | `http://127.0.0.1:11434/v1` | LLM API基础URL（默认Ollama） |
| `--skip-frames` | - | 跳过视频转帧步骤（使用已有帧） |
| `--skip-descriptions` | - | 跳过描述生成步骤（使用已有描述） |

### 单独使用各个模块

#### 1. 视频转帧

```bash
python video_to_frames.py --video data/your_video.mp4 --fps 3 --output ./output
```

#### 2. 生成分组描述

```bash
python frames_to_description.py \
    --frames output/your_video/frames \
    --group-size 10 \
    --overlap 2 \
    --model gemma3:27b
```

## 📁 输出结构

处理完成后，输出目录结构如下：

```
output/
└── your_video/
    ├── frames/              # 视频帧图片
    │   ├── frame_0000.00s.jpg
    │   ├── frame_0000.33s.jpg
    │   └── ...
    ├── group/               # 分组描述JSON
    │   ├── group_1.json
    │   ├── group_2.json
    │   └── ...
    └── summary/             # 总结JSON
        ├── summary_001_0.0s-10.0s.json
        ├── summary_002_10.0s-20.0s.json
        └── ...
```

### JSON文件格式

**group_*.json** - 分组描述文件：
```json
{
  "images": ["frame_0000.00s.jpg", "frame_0000.33s.jpg", ...],
  "description": "视频片段的详细描述..."
}
```

**summary_*.json** - 总结文件：
```json
{
  "time_range": [0.0, 10.0],
  "total_segments": 3,
  "summary": "整合后的完整描述...",
  "segments": [...]
}
```

## 🔧 配置LLM

### 使用Ollama（推荐）

1. 安装并启动 [Ollama](https://ollama.ai/)
2. 下载支持视觉的模型：
   ```bash
   ollama pull gemma3:27b
   # 或其他支持视觉的模型，如 qwen2-vl:7b
   ```
3. 运行脚本时使用默认配置即可

### 使用OpenAI API

```bash
python video_summary_demo.py \
    --video data/your_video.mp4 \
    --api-key your-api-key \
    --model gpt-4-vision-preview \
    --base-url https://api.openai.com/v1
```

### 使用其他兼容OpenAI API的服务

只需修改 `--base-url` 和 `--api-key` 参数即可。

## 📋 依赖要求

- Python >= 3.7
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- openai >= 1.0.0

## 🎯 使用场景

- 📹 视频内容分析和理解
- 📝 自动生成视频字幕和描述
- 🔍 视频内容检索和索引
- 📊 视频内容结构化分析
- 🎓 教育视频内容总结

## 💡 使用技巧

1. **长视频处理**：使用 `--time-window` 参数将长视频分段处理，避免单次处理内容过多
2. **提高精度**：增加 `--fps` 值可以获取更多帧，但会增加处理时间
3. **节省资源**：使用 `--skip-frames` 和 `--skip-descriptions` 可以跳过已完成的步骤
4. **模型选择**：根据视频复杂度选择合适的模型，简单场景可用较小模型

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

本项目使用以下技术：
- [OpenCV](https://opencv.org/) - 视频处理
- [OpenAI Python SDK](https://github.com/openai/openai-python) - LLM API调用
- [Ollama](https://ollama.ai/) - 本地LLM运行

---

**注意**：本项目需要支持视觉理解的多模态大语言模型。确保你使用的模型支持图像输入。

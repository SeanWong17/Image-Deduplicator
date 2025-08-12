# 🤖 高性能图像去重工具 (High-Performance Image Deduplicator)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于深度学习特征和余弦相似度的高性能图像去重工具，能够快速、准确地从海量图像中找出并移除重复或高度相似的图片。

---

## 🚀 主要特性

- **✨ 基于深度学习**：使用预训练的 ResNet50 模型提取图像深层特征，比传统哈希算法（如 pHash）更能识别经过缩放、裁剪、轻微色调变化的相似图像。
- **⚡️ 高性能计算**：利用 PyTorch 和 GPU 加速特征提取；通过分批次的矩阵运算高效计算相似度，轻松处理数万张图片。
- **🔧 灵活可调**：可自由调整相似度阈值，控制去重的严格程度。
- **📁 多线程处理**：并行提取图像特征，充分利用多核 CPU 资源。
- **易于使用**：提供简单的命令行接口，只需几行命令即可完成去重任务。

---

## 🔬 工作流程

本工具的去重流程主要分为两步：

1.  **特征提取 (Feature Extraction)**
    * 使用预训练的 ResNet50 模型（已移除最后的分类层）处理每一张输入图像。
    * 将每张图像转换为一个高维特征向量（2048维）。
    * 最终，所有图像的特征被整合成一个形状为 `(N, 2048)` 的特征矩阵，其中 `N` 是图像总数。

2.  **相似度计算与去重 (Deduplication)**
    * 对特征矩阵进行L2范数归一化，使得后续的点积运算等价于计算 **余弦相似度**。
    * 遍历所有图像，将当前图像与尚未被标记为重复的图像进行比较。
    * 为优化性能，相似度计算采用**分批 (batch)** 方式进行。
    * 若两张图片的特征向量余弦相似度超过预设阈值（如 `0.95`），则将其中一张标记为重复。
    * 最终，输出所有未被标记的、独一无二的图像列表。

---

## 🛠️ 安装指南

1.  **克隆项目**
    ```bash
    git clone [https://github.com/SeanWong17/Image-Deduplicator.git](https://github.com/SeanWong17/Image-Deduplicator.git)
    cd Image-Deduplicator
    ```

2.  **安装依赖**
    建议在一个虚拟环境中安装。PyTorch 的安装请参考其[官方网站](https://pytorch.org/get-started/locally/)，根据您的 CUDA 版本选择合适的命令。
    ```bash
    # 安装 PyTorch (示例，请根据您的环境修改)
    # pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

    # 安装其他依赖
    pip install -r requirements.txt
    ```

---

## 📖 使用方法

本工具提供了一个简单的命令行接口。

**基本用法：**

```bash
python main.py --input-dir /path/to/your/images --output-file unique_images.txt
```

**参数说明：**

* `--input-dir` (必须): 包含待去重图像的文件夹路径。程序会自动扫描该目录下的所有子文件夹。
* `--output-file` (必须): 用于保存去重后唯一图片路径列表的文本文件名。
* `--threshold` (可选): 相似度阈值，介于 0 和 1 之间。值越高，去重越严格。默认为 `0.95`。
* `--gpu-id` (可选): 指定使用的 GPU ID。如果设置为 `-1` 或无可用 GPU，则自动使用 CPU。默认为 `0`。
* `--threads` (可选): 用于特征提取的线程数。默认为 `8`。

**示例：**
```bash
# 对 'data/my_dataset' 文件夹下的图片去重，相似度阈值为0.98，结果保存到 results.txt
python main.py --input-dir data/my_dataset --output-file results.txt --threshold 0.98
```

---

## 作为一个库使用

您也可以在自己的项目中直接导入 `ImageDeduplicator` 类：

```python
from deduplicator import ImageDeduplicator

# 1. 初始化去重器
deduplicator = ImageDeduplicator(gpu_id=0)

# 2. 准备图像路径列表
image_paths = ["img1.jpg", "img2.jpg", "img3.png", ...]

# 3. 执行去重
unique_image_paths = deduplicator.deduplicate(image_paths, threshold=0.95)

# 4. 打印结果
print(f"去重前共有 {len(image_paths)} 张图片。")
print(f"去重后剩余 {len(unique_image_paths)} 张唯一图片。")
for path in unique_image_paths:
    print(path)
```
---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

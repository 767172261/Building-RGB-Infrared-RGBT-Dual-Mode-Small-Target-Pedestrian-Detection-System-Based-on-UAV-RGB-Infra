无人机视角下RGB+红外对齐行人小目标检测数据集
模态与视角：无人机搭载 RGBT 双光相机，从 50–80 m 高度、45°–60° 俯视角采集，同步 RGB + 热红外图像对。
规模：6,125 对图像（4,900 train / 1,225 test），分辨率 640×512，共 70,880 个行人实例。
任务：专门面向 tiny person detection 的无人机 RGBT 检测 benchmark。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1e5e9a7028af40c49b05542edbb3e733.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/95c380f2101f482ca7f50e0db80f0ae0.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/95f335b37d114c6aa21ce7fb06fc223e.png)
以下是 **无人机视角下 RGB+红外对齐行人小目标检测数据集** 的详细信息整理成表格：

---

### 📊 无人机 RGBT 行人小目标检测数据集概览表

| 项目 | 内容 |
|------|------|
| **数据集名称** | DroneRGBT-Pedestrian（或自定义命名） |
| **任务类型** | 小目标行人检测（Tiny Person Detection） |
| **应用场景** | 低空安防、应急搜救、智慧城市、边境巡检、夜间监控 |
| **采集平台** | 无人机（UAV）搭载 **RGB + 热红外（Thermal）双光相机** |
| **模态** | **双模态同步图像对**：<br>• 可见光（RGB）<br>• 热红外（Infrared / Thermal） |
| **采集高度** | 50 – 80 米 |
| **俯视角** | 45° – 60° 斜向俯拍 |
| **图像分辨率** | **640 × 512 像素**（RGB 与红外严格对齐） |
| **图像对数量** | **6,125 对**（每对含 1 张 RGB + 1 张红外图） |
| **数据划分** | - **训练集（train）**：4,900 对<br>- **测试集（test）**：1,225 对 |
| **标注格式** | 支持 YOLO / COCO / VOC 等（通常提供边界框 `.txt` 或 `.json`） |
| **目标类别** | **1 类：行人（Person）** |
| **总实例数** | **70,880 个行人标注框**（平均每个图像约 11.6 个目标） |
| **目标特点** | - **小目标为主**（多数 < 32×32 像素）<br>- 包含遮挡、密集、远距离、夜间/低照度场景<br>- 红外模态增强弱光/夜间检测能力 |
| **对齐方式** | **像素级空间对齐**（经标定与配准，RGB 与红外图一一对应） |
| **适用模型** | - 单模态：YOLOv8（RGB 或 Thermal）<br>- 多模态融合：Fusion-YOLO、CFT、MFFNet、GLAD 等 RGBT 检测模型 |
| **用途** | • 无人机小目标检测 benchmark<br>• 多模态特征融合研究<br>• 跨模态对齐与迁移学习 |

---

该数据集专为 **挑战性极高的无人机小行人检测任务** 设计，特别适合评估模型在 **远距离、小尺度、复杂光照（含夜间）** 条件下的鲁棒性。

以下是构建 **基于无人机 RGB+红外（RGBT）双模态小目标行人检测系统** 的完整代码方案，采用 **YOLOv8 多模态融合架构**（简化版），支持训练、推理与可视化。由于 Ultralytics YOLOv8 原生不直接支持双模态输入，我们将通过 **早期融合（Early Fusion）** 方式将 RGB 与红外图拼接为 4 通道输入，并微调模型。

> 💡 注：本方案适用于科研/工程落地，可作为 benchmark 系统基础。

---

## 📁 一、项目结构

```bash
DroneRGBT_Detection/
├── datasets/
│   ├── images/
│   │   ├── train/          # 存放融合后的 4 通道 .npy 或拼接图
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   └── yolov8s-rgbt.yaml   # 自定义 4 通道输入模型配置
├── tools/
│   ├── fuse_rgb_thermal.py # 将 RGB + 红外融合为 4 通道图像
│   └── create_labels.py    # （若需从原始标注生成 YOLO 格式）
├── train.py                # 训练脚本
├── detect.py               # 推理脚本（支持单图/视频）
└── rgbt_drone.yaml         # 数据集配置文件
```

---

## 🔧 二、1. 融合 RGB + 红外图像（`fuse_rgb_thermal.py`）

```python
# tools/fuse_rgb_thermal.py
import os
import cv2
import numpy as np

def fuse_rgb_thermal(rgb_dir, thermal_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = sorted(os.listdir(rgb_dir))
    
    for rgb_file in rgb_files:
        if not rgb_file.lower().endswith(('.jpg', '.png')):
            continue
        thermal_file = rgb_file  # 假设文件名一致
        
        rgb_path = os.path.join(rgb_dir, rgb_file)
        thermal_path = os.path.join(thermal_dir, thermal_file)
        
        rgb = cv2.imread(rgb_path)  # (H, W, 3)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
        
        if rgb is None or thermal is None:
            print(f"Skip {rgb_file}")
            continue
            
        # 调整 thermal 到 3 通道（可选）或保留单通道
        thermal = np.expand_dims(thermal, axis=2)  # (H, W, 1)
        fused = np.concatenate([rgb, thermal], axis=2)  # (H, W, 4)
        
        # 保存为 .npy（推荐）或拼接图（如用 4 通道 TIFF）
        np.save(os.path.join(output_dir, rgb_file.replace('.jpg', '.npy')), fused)

# 示例使用
if __name__ == '__main__':
    fuse_rgb_thermal(
        rgb_dir='raw_data/train/rgb',
        thermal_dir='raw_data/train/thermal',
        output_dir='datasets/images/train'
    )
```

> ✅ 输出为 `.npy` 文件（含 4 通道），便于后续加载。

---

## 🛠️ 三、2. 自定义 YOLOv8 模型（4 通道输入）

创建 `models/yolov8s-rgbt.yaml`：

```yaml
# YOLOv8s for RGBT (4-channel input)
nc: 1  # number of classes
scales: [0.33, 0.50]  # model depth and width scaling

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2  (input must be 4-channel!)
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

> ⚠️ 关键修改：首层 `Conv` 输入通道自动适配 4 通道（无需改代码，PyTorch 自动推断）。

---

## 📂 四、3. 数据集配置（`rgbt_drone.yaml`）

```yaml
# rgbt_drone.yaml
train: ./datasets/images/train  # 实际存放 .npy 路径
val: ./datasets/images/val

nc: 1
names: ['person']
```

> ❗ 注意：Ultralytics 默认读取图像文件（.jpg/.png），但我们需要加载 `.npy`。因此需自定义数据加载器。

---

## 🧩 五、4. 自定义数据加载器（关键！）

在 `train.py` 中重写 `load_image` 行为（简化版）：

```python
# train.py
from ultralytics import YOLO
import torch
import numpy as np
import os
from pathlib import Path

# 替换默认的图像加载方式
def custom_load_image(self, i):
    """Load 4-channel .npy image"""
    path = self.files[i]
    if path.endswith('.npy'):
        img = np.load(path)  # (H, W, 4)
        img = img.transpose(2, 0, 1)  # (4, H, W)
        img = torch.from_numpy(img).float()
        return img, img.shape[1], img.shape[2]
    else:
        # fallback
        return self._old_load_image(i)

# Monkey patch（仅用于训练）
from ultralytics.data.dataset import YOLODataset
YOLODataset._old_load_image = YOLODataset.load_image
YOLODataset.load_image = custom_load_image

# 修改标签加载（确保 .txt 对应 .npy）
def custom_get_label_file(self, img_path):
    return str(Path(img_path).with_suffix('.txt'))

YOLODataset.get_label_file = custom_get_label_file
```

> 更稳健做法：继承 `YOLODataset` 并注册新类，但上述 monkey patch 可快速验证。

---

## ▶️ 六、5. 完整训练脚本（`train.py`）

```python
# train.py（完整版）
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path

# === 自定义数据加载（支持 .npy）===
from ultralytics.data.dataset import YOLODataset

def load_image_npy(self, i):
    f = self.im_files[i]
    if f.endswith('.npy'):
        im = np.load(f)  # (H, W, 4)
        im = im.transpose(2, 0, 1)  # (4, H, W)
        im = torch.from_numpy(im).float()
        h, w = im.shape[1], im.shape[2]
        return im, h, w
    else:
        return self._orig_load_image(i)

def get_label_file_npy(self, img_path):
    return str(Path(img_path).with_suffix('.txt'))

# Patch
YOLODataset._orig_load_image = YOLODataset.load_image
YOLODataset.load_image = load_image_npy
YOLODataset.get_label_file = get_label_file_npy

# === 开始训练 ===
if __name__ == '__main__':
    model = YOLO('models/yolov8s-rgbt.yaml')  # 从头训练
    # 或加载预训练权重（需修改首层 conv1.weight）
    # model = YOLO('yolov8s.pt')
    # model.model.model[0].conv = torch.nn.Conv2d(4, 64, 3, 2, 1)  # 手动替换

    model.train(
        data='rgbt_drone.yaml',
        epochs=100,
        imgsz=512,
        batch=16,
        name='drone_rgbdet',
        project='runs',
        device=0,
        cache=False,
        workers=4
    )
```

---

## 🔍 七、6. 推理脚本（`detect.py`）

```python
# detect.py
import torch
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO('runs/drone_rgbdet/weights/best.pt')

def detect_rgbd_pair(rgb_path, thermal_path):
    rgb = cv2.imread(rgb_path)  # (H, W, 3)
    thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
    thermal = np.expand_dims(thermal, axis=2)
    fused = np.concatenate([rgb, thermal], axis=2)  # (H, W, 4)
    fused = fused.transpose(2, 0, 1)  # (4, H, W)
    fused = torch.from_numpy(fused).float().unsqueeze(0) / 255.0  # 归一化

    results = model(fused, augment=True)
    # 可视化：叠加在 RGB 图上
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
    cv2.waitKey(0)

# 示例
detect_rgbd_pair('test/rgb/001.jpg', 'test/thermal/001.jpg')
```

---

## ✅ 八、部署建议

| 方案 | 说明 |
|------|------|
| **科研实验** | 使用 `.npy` + 自定义 DataLoader |
| **工程部署** | 将 RGB + 红外拼接为 4 通道 TIFF / PNG（需扩展 OpenCV 支持） |
| **加速推理** | 导出 ONNX 后使用 TensorRT（需处理 4 通道输入） |
| **替代融合策略** | 中期融合（Two-Stream + 特征拼接）效果更佳，但需重写模型 |

---



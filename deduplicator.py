import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class ImageDeduplicator:
    def __init__(self, model_name='resnet50', num_threads=8, gpu_id=0):
        """
        初始化去重器
        :param model_name: 使用的预训练模型名称（目前仅支持 resnet50）
        :param num_threads: 多线程的线程数
        :param gpu_id: 指定使用的 GPU ID。如果设置为 -1，则使用 CPU。
        """
        self.device = self._set_device(gpu_id)
        self.model = self._load_model(model_name).to(self.device).eval()
        self.num_threads = num_threads
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _set_device(self, gpu_id):
        """设置设备（GPU 或 CPU）"""
        if gpu_id == -1 or not torch.cuda.is_available():
            print("Using CPU for computation.")
            return torch.device("cpu")
        
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # This is better set before launching the script
        print(f"Using GPU: {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")

    def _load_model(self, model_name):
        """加载预训练模型，并移除分类层"""
        if model_name != 'resnet50':
            raise ValueError(f"Unsupported model: {model_name}")
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        return torch.nn.Sequential(*list(model.children())[:-1])

    def _extract_feature(self, image_path):
        """提取单张图像的特征向量"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature_vector = self.model(image_tensor).squeeze().cpu().numpy()
                return feature_vector
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_features(self, image_paths):
        """使用多线程提取图像特征"""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(tqdm(executor.map(self._extract_feature, image_paths), total=len(image_paths), desc="Extracting features"))
        
        valid_features = [res for res in results if res is not None]
        valid_indices = [i for i, res in enumerate(results) if res is not None]
        
        if not valid_features:
            return np.array([]), []
            
        return np.array(valid_features), valid_indices

    def deduplicate_by_similarity(self, features, threshold=0.95, batch_size=1000):
        """
        基于相似度进行去重
        :param features: 特征矩阵 (N x D)
        :param threshold: 相似度阈值
        :param batch_size: 批处理大小
        :return: 保留的图像索引列表
        """
        num_images = features.shape[0]
        if num_images == 0:
            return []
            
        # 标准化特征向量 (L2 norm)
        features /= np.linalg.norm(features, axis=1, keepdims=True)

        to_keep = []
        excluded = np.zeros(num_images, dtype=bool)

        for i in tqdm(range(num_images), desc="Deduplicating", unit="image"):
            if excluded[i]:
                continue
            
            to_keep.append(i)
            # Find indices of images that are not yet excluded
            remaining_indices = np.where(~excluded[i + 1:])[0] + i + 1
            if len(remaining_indices) == 0:
                break

            # Process in batches
            for j in range(0, len(remaining_indices), batch_size):
                batch_indices = remaining_indices[j:j + batch_size]
                batch_features = features[batch_indices]
                
                # Calculate cosine similarity
                similarities = np.dot(features[i:i+1], batch_features.T).squeeze()
                
                # Mark duplicates
                duplicate_mask = similarities > threshold
                excluded[batch_indices[duplicate_mask]] = True
        
        return to_keep

    def deduplicate(self, image_paths, threshold=0.95, batch_size=1000):
        """
        对图像路径列表去重
        :return: 去重后的图像路径列表
        """
        print(f"Found {len(image_paths)} images to process.")
        
        features, valid_indices = self.extract_features(image_paths)
        if features.shape[0] == 0:
            print("No valid features extracted. Exiting.")
            return []
            
        original_valid_paths = [image_paths[i] for i in valid_indices]
        
        indices_to_keep = self.deduplicate_by_similarity(features, threshold, batch_size)
        
        unique_paths = [original_valid_paths[i] for i in indices_to_keep]
        return unique_paths

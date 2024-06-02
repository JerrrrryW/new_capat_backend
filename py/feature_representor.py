import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# 定义图像预处理流程
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()  # 设置为评估模式

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        # 获取模型的最后一个全连接层的输出特征
        output = model(batch_t)
        return output.cpu().numpy().flatten()

def load_images_from_folder(folder):
    features_list = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            try:
                features = extract_features(img_path)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return np.array(features_list)

# 指定图像文件夹路径
folder_path = 'fragments'

# 从文件夹加载图像并提取特征矩阵
features_matrix = load_images_from_folder(folder_path)

# 显示每个图像的特征矩阵
for i, features in enumerate(features_matrix):
    print(f"Image {i} features: {features.shape}")

# 使用t-SNE算法对特征矩阵降维至1维
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 假设time_stamps是时间戳数组，features是对应的图片特征矩阵
# 根据特征矩阵的行数生成时间戳数组
time_stamps = np.arange(features_matrix.shape[0])

# 使用t-SNE将数据降维到一维
tsne = TSNE(n_components=1, random_state=42) # perplexity是t-SNE的一个参数，用于控制降维后的分布
tsne_results = tsne.fit_transform(features_matrix)

print(f"t-SNE 1D results: {tsne_results}")

# # 绘制散点图，纵坐标是t-SNE的输出
# plt.scatter(time_stamps, tsne_results, alpha=0.5)
# plt.xlabel('Time')
# plt.ylabel('t-SNE 1D')
# plt.title('t-SNE Reduction to 1D')
# plt.show()
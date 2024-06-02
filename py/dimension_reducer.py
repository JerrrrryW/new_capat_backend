# 使用t-SNE算法对特征矩阵降维至1维
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 假设time_stamps是时间戳数组，features是对应的图片特征矩阵
# 根据特征矩阵的行数生成时间戳数组
time_stamps = np.arange(features_matrix.shape[0])

# 使用t-SNE将数据降维到一维
tsne = TSNE(n_components=1, random_state=42, perplexity=2) # perplexity是t-SNE的一个参数，用于控制降维后的分布
tsne_results = tsne.fit_transform(features_matrix)

print(f"t-SNE 1D results: {tsne_results}")

# # 绘制散点图，纵坐标是t-SNE的输出
# plt.scatter(time_stamps, tsne_results, alpha=0.5)
# plt.xlabel('Time')
# plt.ylabel('t-SNE 1D')
# plt.title('t-SNE Reduction to 1D')
# plt.show()
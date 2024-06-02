import pickle
import pymysql
import numpy as np
from sklearn.manifold import TSNE
import json
import random

# 读取特征向量数据
with open('fragments_data.pkl', 'rb') as file:
    data = pickle.load(file)

# 计算每个画作的平均特征向量
average_feature_vectors = {}
fragments_count = {}
for fragment in data:
    pid = fragment['filename'].split("_")[0]  # 获取绘画id
    if pid not in average_feature_vectors:
        average_feature_vectors[pid] = []
        fragments_count[pid] = 0
    average_feature_vectors[pid].append(fragment['features'])
    fragments_count[pid] += 1
    

# 计算平均值
for pid in average_feature_vectors.keys():
    average_feature_vectors[pid] = np.mean(average_feature_vectors[pid], axis=0)

# 使用t-SNE进行降维
features_matrix = np.array(list(average_feature_vectors.values()))
tsne = TSNE(n_components=1)
tsne_results = tsne.fit_transform(features_matrix)

# 准备生成JSON
paintings_info = []

# 连接数据库
try:
    db = pymysql.connect(host='localhost', user='root', passwd='DAXI9999', port=3306, db='capat_labelling')
    cursor = db.cursor()
    print('数据库连接成功！')
except Exception as e:
    print('数据库连接失败：', str(e))
    exit()

# 对每个画作查询其基本信息并结合降维结果
for i, pid in enumerate(average_feature_vectors.keys()):
    sql = f"SELECT Paintings.PID, author_name, title, era, year, collection_institution, material, dimensions " \
          f"FROM Paintings WHERE PID = '{pid}';"
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        if result:
             # 处理年份信息
            if result[4] is not None:  # 如果year不是null
                year = int(result[4])
            else:  # 如果year是null，随机生成一个年份
                year = random.randint(1642, 1700)  
            painting_info = {
                'title': result[2],
                'dynasty': result[3],
                'year': str(year),
                'PID': result[0],
                'author_name': result[1],
                'fragments_count': fragments_count[pid],  # 片段数量
                'collection_institution': result[5],
                'material': result[6],
                'dimensions': result[7],
                'tsne_1d': tsne_results[i][0].item(),  # 转换为Python原生类型以便JSON序列化
                
            }
            paintings_info.append(painting_info)
    except Exception as e:
        print("查询失败：", str(e))

# 关闭数据库连接
db.close()

# 保存到JSON文件
with open('paintings_tsne.json', 'w') as outfile:
    json.dump(paintings_info, outfile)

print("完成。JSON文件已生成。")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Đọc dữ liệu
data = pd.read_csv('results.csv')

# 2. Tiền xử lý dữ liệu
# Loại bỏ cột text nếu có (Nation, Team, Position)
data_numeric = data.select_dtypes(include=[np.number])

# Xử lý giá trị thiếu (nếu có)
data_numeric = data_numeric.fillna(0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)

# 3. Áp dụng K-means với K=4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Thêm nhãn cụm vào dữ liệu gốc
data['Cluster'] = labels

# 4. Giảm chiều dữ liệu bằng PCA (2D)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Thêm PCA vào dataframe để tiện vẽ
pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = labels

# 5. Vẽ biểu đồ scatter 2D
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green', 'orange']
for cluster in range(4):
    cluster_points = pca_df[pca_df['Cluster'] == cluster]
    plt.scatter(cluster_points['PCA1'], cluster_points['PCA2'], 
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)

plt.title('2D PCA clusters visualization of Players')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.grid(True)
plt.show()

# 6. (Tùy chọn) Xuất file kết quả
data.to_csv('results_with_clusters.csv', index=False)

print("Đã phân cụm xong, số lượng cầu thủ mỗi cụm:")
print(data['Cluster'].value_counts())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Đọc dữ liệu
data = pd.read_csv('results.csv')

# Tiền xử lý dữ liệu
# Loại bỏ cột text nếu có (Nation, Team, Position)
data_numeric = data.select_dtypes(include=[np.number])

# Xử lý giá trị thiếu (nếu có)
data_numeric = data_numeric.fillna(0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)

# Vẽ biểu đồ Elbow để chọn số cụm tối ưu 
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(k_range, wcss, 'o-', marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

# Áp dụng K-means với K=4 
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Thêm nhãn cụm vào dữ liệu gốc
data['Cluster'] = labels

# Giảm chiều dữ liệu bằng PCA (2D)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Thêm PCA vào dataframe để tiện vẽ
pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = labels

# Vẽ biểu đồ scatter 2D
plt.figure(figsize=(12,6))
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

# (Tùy chọn) Xuất file kết quả
data.to_csv('results_with_clusters.csv', index=False)

print("Đã phân cụm xong, số lượng cầu thủ mỗi cụm:")
print(data['Cluster'].value_counts())

# --- Giải thích ý nghĩa các nhóm (Cluster) ---
# Cluster 0: Cầu thủ có khả năng ghi bàn cao (Goals cao)
# Cluster 1: Cầu thủ hỗ trợ tốt (Assists cao, tiền vệ sáng tạo)
# Cluster 2: Cầu thủ phòng ngự (Tackles, Clearances cao)
# Cluster 3: Cầu thủ toàn diện (Interceptions, Passes mạnh)

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def prepare_datasource_metadata(embeddings):
    """
    Tính toán các đặc trưng tĩnh cho một nguồn dữ liệu (data source).
    :param embeddings: mảng numpy (N, D) chứa các vector nhúng của tài liệu.
    :return: dict chứa centroid, density, và số lượng item.
    """
    # 1. Số lượng item (Feature iv)
    num_items = len(embeddings)
    
    # 2. Centroid (Feature ii): Trung bình cộng của các vector
    centroid = np.mean(embeddings, axis=0)
    
    # 3. Density (Feature v): Độ đậm đặc
    # Cách tính: Trung bình khoảng cách từ các vector đến centroid
    # Một nguồn dữ liệu có density cao nếu các vector nằm rất gần centroid.
    distances = euclidean_distances(embeddings, centroid.reshape(1, -1))
    avg_distance = np.mean(distances)
    
    # Density tỷ lệ nghịch với khoảng cách trung bình
    density = 1.0 / (avg_distance + 1e-6) 
    
    return {
        "centroid": centroid,
        "num_items": num_items,
        "density": density
    }

# Ví dụ thực hiện cho 3 nguồn dữ liệu
# data_sources = [embeddings_source_1, embeddings_source_2, ...]
# metadata_store = [prepare_datasource_metadata(ds) for ds in data_sources]
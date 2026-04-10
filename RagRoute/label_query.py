import faiss
import numpy as np

def generate_labels_for_query(query_eb, indices, k=10):
    """
    Gán nhãn binary cho từng nguồn dữ liệu dựa trên Top-k Global.
    :param query_eb: Vector nhúng của câu hỏi (1, D)
    :param indices: Danh sách các FAISS indices của n nguồn
    :param k: Số lượng tài liệu k được chọn cho RAG
    :return: list nhãn [0, 1, 0...] cho n nguồn
    """
    all_candidate_results = []
    
    # 1. Thu thập ứng viên từ TẤT CẢ các nguồn
    for i, index in enumerate(indices):
        # Lấy top-k của riêng nguồn đó
        distances, _ = index.search(query_eb.astype('float32'), k)
        
        # Lưu lại khoảng cách và nguồn gốc
        for dist in distances[0]:
            all_candidate_results.append({
                'dist': dist,
                'source_id': i
            })
            
    # 2. Bước REFINE: Sắp xếp lại toàn bộ ứng viên theo khoảng cách L2
    # Khoảng cách càng nhỏ (L2) thì càng liên quan
    all_candidate_results.sort(key=lambda x: x['dist'])
    
    # 3. Lọc lấy Top-k thực tế trên toàn hệ thống
    global_top_k = all_candidate_results[:k]
    
    # 4. Xác định nguồn nào 'đóng góp' tài liệu vào Top-k Global này
    relevant_source_indices = {item['source_id'] for item in global_top_k}
    
    # 5. Tạo vector nhãn binary
    num_sources = len(indices)
    labels = [1 if i in relevant_source_indices else 0 for i in range(num_sources)]
    
    return labels
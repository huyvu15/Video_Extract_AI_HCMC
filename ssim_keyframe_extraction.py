import cv2
import numpy as np

def extract_keyframes(video_path, threshold=25):
    cap = cv2.VideoCapture(video_path)
    
    # Đọc frame đầu tiên
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    keyframes = [prev_frame]
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(frame_gray, prev_frame_gray)
        
        # Tính giá trị trung bình của sự khác biệt
        mean_diff = np.mean(diff)
        
        # Nếu sự khác biệt vượt quá ngưỡng, lưu frame là keyframe
        if mean_diff > threshold:
            keyframes.append(frame)
        
        prev_frame_gray = frame_gray
        frame_count += 1

    cap.release()
    return keyframes

# Sử dụng hàm
# video_path = "taxi-TQ.mp4"
video_path = 'AOE.mp4'

keyframes = extract_keyframes(video_path)

# Lưu các keyframe
for i, frame in enumerate(keyframes):
    cv2.imwrite(f"keyframe_{i}.jpg", frame)

print(f"Đã trích xuất {len(keyframes)} keyframe.")
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# def evaluate_keyframes(video_path, keyframes):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Tính tỷ lệ nén
#     compression_ratio = len(keyframes) / total_frames
#     print(f"Tỷ lệ nén: {compression_ratio:.2f}")
    
#     # Tính độ đa dạng của keyframes
#     diversity = []
#     for i in range(1, len(keyframes)):
#         prev_frame = cv2.cvtColor(keyframes[i-1], cv2.COLOR_BGR2GRAY)
#         curr_frame = cv2.cvtColor(keyframes[i], cv2.COLOR_BGR2GRAY)
#         similarity = ssim(prev_frame, curr_frame)
#         diversity.append(1 - similarity)
    
#     avg_diversity = np.mean(diversity)
#     print(f"Độ đa dạng trung bình của keyframes: {avg_diversity:.2f}")
    
#     # Kiểm tra độ bao phủ
#     frame_indices = []
#     for i, frame in enumerate(keyframes):
#         while True:
#             ret, video_frame = cap.read()
#             if not ret:
#                 break
#             if np.array_equal(frame, video_frame):
#                 frame_indices.append(i)
#                 break
    
#     coverage = len(set(frame_indices)) / total_frames
#     print(f"Độ bao phủ của keyframes: {coverage:.2f}")
    
#     cap.release()

# # Sử dụng hàm đánh giá
# video_path = "taxi-TQ.mp4"
# keyframes = extract_keyframes(video_path)
# evaluate_keyframes(video_path, keyframes)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# def extract_keyframes(video_path, threshold):
#     cap = cv2.VideoCapture(video_path)
#     keyframes = []
#     prev_frame = None
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         if prev_frame is None:
#             prev_frame = frame_gray
#             keyframes.append(frame)
#             continue
        
#         diff = cv2.absdiff(frame_gray, prev_frame)
#         mean_diff = np.mean(diff)
        
#         if mean_diff > threshold:
#             keyframes.append(frame)
        
#         prev_frame = frame_gray
#         frame_count += 1
    
#     cap.release()
#     return keyframes, frame_count

# def evaluate_keyframes(keyframes, total_frames):
#     compression_ratio = len(keyframes) / total_frames
    
#     diversity = []
#     for i in range(1, len(keyframes)):
#         prev_frame = cv2.cvtColor(keyframes[i-1], cv2.COLOR_BGR2GRAY)
#         curr_frame = cv2.cvtColor(keyframes[i], cv2.COLOR_BGR2GRAY)
#         similarity = ssim(prev_frame, curr_frame)
#         diversity.append(1 - similarity)
    
#     avg_diversity = np.mean(diversity) if diversity else 0
    
#     return compression_ratio, avg_diversity

# def find_optimal_threshold(video_path, start=10, end=100, step=5):
#     results = []
    
#     for threshold in range(start, end, step):
#         keyframes, total_frames = extract_keyframes(video_path, threshold)
#         compression_ratio, diversity = evaluate_keyframes(keyframes, total_frames)
        
#         results.append({
#             'threshold': threshold,
#             'compression_ratio': compression_ratio,
#             'diversity': diversity,
#             'num_keyframes': len(keyframes)
#         })
    
#     return results

# def plot_results(results):
#     thresholds = [r['threshold'] for r in results]
#     compression_ratios = [r['compression_ratio'] for r in results]
#     diversities = [r['diversity'] for r in results]
#     num_keyframes = [r['num_keyframes'] for r in results]
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
#     ax1.plot(thresholds, compression_ratios, label='Tỷ lệ nén')
#     ax1.plot(thresholds, diversities, label='Độ đa dạng')
#     ax1.set_xlabel('Ngưỡng')
#     ax1.set_ylabel('Giá trị')
#     ax1.set_title('Ảnh hưởng của ngưỡng đến tỷ lệ nén và độ đa dạng')
#     ax1.legend()
    
#     ax2.plot(thresholds, num_keyframes)
#     ax2.set_xlabel('Ngưỡng')
#     ax2.set_ylabel('Số lượng keyframe')
#     ax2.set_title('Ảnh hưởng của ngưỡng đến số lượng keyframe')
    
#     plt.tight_layout()
#     plt.show()

# def find_best_threshold(results):
#     # Chuẩn hóa các chỉ số
#     max_compression = max(r['compression_ratio'] for r in results)
#     max_diversity = max(r['diversity'] for r in results)
    
#     for r in results:
#         r['normalized_compression'] = r['compression_ratio'] / max_compression
#         r['normalized_diversity'] = r['diversity'] / max_diversity
#         r['score'] = r['normalized_compression'] * r['normalized_diversity']
    
#     best_result = max(results, key=lambda x: x['score'])
#     return best_result['threshold']

# # Sử dụng các hàm
# video_path = "taxi-TQ.mp4"
# results = find_optimal_threshold(video_path)
# plot_results(results)

# best_threshold = find_best_threshold(results)
# print(f"Ngưỡng tốt nhất: {best_threshold}")

# # Trích xuất keyframe với ngưỡng tốt nhất
# final_keyframes, total_frames = extract_keyframes(video_path, best_threshold)
# final_compression_ratio, final_diversity = evaluate_keyframes(final_keyframes, total_frames)

# print(f"Số lượng keyframe: {len(final_keyframes)}")
# print(f"Tỷ lệ nén cuối cùng: {final_compression_ratio:.4f}")
# print(f"Độ đa dạng cuối cùng: {final_diversity:.4f}")
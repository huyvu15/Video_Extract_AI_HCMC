import cv2
import numpy as np
import os

def extract_video_segments_with_keyframes(video_path, threshold=30, segment_length=30):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy số khung hình trên giây
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Đọc frame đầu tiên
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    video_segments = []
    current_segment = []
    segment_start = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(frame_gray, prev_frame_gray)
        
        # Tính giá trị trung bình của sự khác biệt
        mean_diff = np.mean(diff)
        
        # Nếu sự khác biệt vượt quá ngưỡng, lưu frame là keyframe và bắt đầu một đoạn video mới
        if mean_diff > threshold:
            if current_segment:
                video_segments.append(current_segment)
            current_segment = []
            segment_start = frame_count
        
        current_segment.append(frame)
        
        # Đảm bảo rằng đoạn video có độ dài nhất định
        if len(current_segment) > segment_length * fps:
            video_segments.append(current_segment)
            current_segment = []

        prev_frame_gray = frame_gray
        frame_count += 1

    if current_segment:
        video_segments.append(current_segment)

    cap.release()
    return video_segments, width, height, fps

# Sử dụng hàm để trích xuất các đoạn video
video_path = 'taxi-TQ.mp4'
video_segments, width, height, fps = extract_video_segments_with_keyframes(video_path)

# Lưu các đoạn video
output_dir = 'output_segments1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, segment in enumerate(video_segments):
    segment_path = os.path.join(output_dir, f'segment_{i}.avi')
    out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    for frame in segment:
        out.write(frame)
    out.release()

print(f"Đã trích xuất {len(video_segments)} đoạn video chứa keyframe.")

import cv2
import os

def extract_keyframes(video_path, output_dir, threshold=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    success, prev_frame = cap.read()
    
    if not success:
        print("Không thể mở video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    keyframes = [prev_frame]
    
    while success:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        non_zero_count = cv2.countNonZero(diff)
        
        if non_zero_count > threshold * 1000:
            keyframes.append(frame)
            prev_gray = gray
            keyframe_path = os.path.join(output_dir, f"keyframe_{frame_id}.jpg")
            cv2.imwrite(keyframe_path, frame)
            print(f"Đã lưu keyframe tại: {keyframe_path}")

        frame_id += 1

    cap.release()
    print(f"Tổng số keyframes trích xuất: {len(keyframes)}")

if __name__ == "__main__":
    video_path = "taxi-TQ.mp4"
    output_dir = "keyframes_output"
    extract_keyframes(video_path, output_dir)

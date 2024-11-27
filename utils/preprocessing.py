import cv2
import os

def get_img(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    count = 0
    frame_count = 0
    video_path = video_path.replace('.mp4', '')
    path_img = '/'.join(video_path.split('/')[-2:])
    dir = f'img_origin/{path_img}'
    sub1 = video_path.split('/')[-2]
    sub2 = video_path.split('/')[-1].split('_')[-4]
    if not os.path.exists(dir):
        os.makedirs(dir)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_filename = f"{dir}/{sub1}_{sub2}_frame_{count}.jpg"
                cv2.imwrite(frame_filename, gray)
                count += 1
            frame_count += 1
        else:
            print("Error")
            break
import cv2
import os 

videos_dir = "./training/videos"
frames_dir = "./training/frames"

if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

videos_list = sorted(os.listdir(videos_dir))
print(videos_list)

for video in videos_list:
    save_path = os.path.join(frames_dir, video.split(".")[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载video
    cap = cv2.VideoCapture(os.path.join(videos_dir, video))
    isOpened = cap.isOpened()
    index = 0
    while isOpened:
        flag, frame = cap.read()
        filename = str(index).zfill(4) + ".jpg"
        if flag == True:
            cv2.imwrite(os.path.join(save_path,filename), frame)
        else:
            break
        index = index + 1
    cap.release()
    print("end ", video)

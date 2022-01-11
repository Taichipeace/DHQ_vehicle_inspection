import cv2

# 设置摄像头编号
camera_id = '34020000001320001011'

# 设置视频流开始时间
time_start = '20220106154740'

# 设置视频流结束时间
time_end = '20220106154815'

# 设置要保存图像的间隔, frame_eval为每frame_eval帧保存一张图像
frame_eval = 3

# 获取视频流的地址
strm_url = f"http://dxlbs.smart-ideas.com.cn:10000/api/v1/cloudrecord/video/play/" \
                f"{camera_id}/{camera_id}/{time_start}/{time_end}/video.mp4"
print(strm_url)

# 设置图片保存路径
output_path = 'D:/workpath/dataset_images/'

strm = cv2.VideoCapture(strm_url)
ret = 1
i = 0
while ret:
    ret, frame = strm.read()
    if i % frame_eval == 0:
        img_full_path = f'{output_path}{camera_id}_{time_start}_{time_end}_{i}.jpg'
        cv2.imwrite(img_full_path, frame)
        print(f'{camera_id}_{time_start}_{time_end}_{i}.jpg  is saved')
    i = i + 1
strm.release()

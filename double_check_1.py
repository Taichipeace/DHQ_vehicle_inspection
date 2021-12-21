# ATAI 更新日志 2021.12.12 10：58
# 1. 增加二次检查后，check_res = ‘1’的数据推送和文件移动。
# 2. 检测致信值阈值调高至 0.65

import requests
# import time
# import datetime
import ppdet
from TedTools import *


def get_check_imgs(strm_prefix, t0_full):
    t0 = t0_full.split('/')[-1]
    ip_name = t0_full.split('/')[-2]
    time_start = shift_time(t0, -15)
    time_end = shift_time(t0, 15)
    strm_url = strm_prefix + time_start[:14] + '/' + time_end[:14] + '/video.mp4'
    while True:
        strm = cv2.VideoCapture(strm_url)
        # 判断视频是否正确打开
        if strm.isOpened():
            break
        else:
            print(f"所需的二次检查视频尚未生成，等待5秒后自动重试: {time_start}")
            time.sleep(5)
    strm_fps = strm.get(cv2.CAP_PROP_FPS)
    ret, frame = strm.read()
    i = 0
    frame_eval = int(strm_fps * 2)  # 设置要保存图像的间隔 2为每隔2秒保存一张图像
    while ret:
        # 如果视频仍然存在，继续创建图像
        if i % frame_eval == 0:
            img_full_path = t0_full + '/' + ip_name + '_01_' + shift_time(time_start, (i / frame_eval)*2)  + '_VEHICLE_DETECTION.jpg'
            cv2.imwrite(img_full_path, frame)
        ret, frame = strm.read()
        i = i + 1
    strm.release()
    cv2.destroyAllWindows()


def get_stream_url(strm_prefix, t0_full):
    t0 = t0_full.split('/')[-1]
    ip_name = t0_full.split('/')[-2]
    time_start = shift_time(t0, -15)
    time_end = shift_time(t0, 15)
    strm_url = strm_prefix + time_start[:14] + '/' + time_end[:14] + '/video.mp4'
    return strm_url


# 推送数据到数据库的开关
post_trigger = False

# 移动图片的开关
move_file_trigger = True

# 保存二次检查需使用图片的路径
double_check_dir = "D:/workpath/ftp_double_check/"
pathDoneFile = "D:/workpath/http/"

# 推送数据到服务器的地址
post_url = "http://daxing.smart-ideas.com.cn/inputcarocr.php"

# 获取视频流的地址前缀
stream_prefix = "http://dxlbs.smart-ideas.com.cn:10000/api/v1/cloudrecord/video/play/34020000001320001011/34020000001320001011/"

# 初始化人车检测模型
# 读取模型参数
cfg = 'PaddleDetection/configs/picodet/picodet_l_640_coco.yml'
weights = 'TedMod/picodet_l_640_coco.pdparams'

# 模型初始化
cfg = ppdet.core.workspace.load_config(cfg)
trainer = ppdet.engine.Trainer(cfg, mode='test')
trainer.load_weights(weights)

# 设置Detction结果保存路径
output_path = 'PPdetOutput/'

# 读取保存到pkl文件的pts列表
roi_HongDa_in_2nd = read_roi_contour('TedRoiFiles/PolyRoi_192.168.10.11_2nd.pkl')

while True:
    try:
        if not os.path.exists(double_check_dir):  # 如果ftp_double_check文件夹下还没有ip摄像头文件夹
            print(f"IP摄像头文件夹： {double_check_dir} 是空的，等30秒再来")
        else:
            ip_dir_list = os.listdir(double_check_dir)  # 获取ip摄像头文件夹列表
            for ip_dir in ip_dir_list:
                if not os.listdir(double_check_dir+ip_dir):  # 如果192.168.x.x文件夹下还没有t0文件夹
                    print(f"t0文件夹：{double_check_dir+ip_dir} 是空的，等30秒再来")
                else:
                    t0_dir_list = os.listdir(double_check_dir+ip_dir)  # 获取t0文件夹列表
                    for t0_dir in t0_dir_list:
                        img_dir_full = double_check_dir+ip_dir + '/' + t0_dir
                        # stream_url = get_stream_url(stream_prefix, img_dir_full)
                        img_list = os.listdir(img_dir_full)  # 获取t0_rep里的图片列表
                        if len(img_list) == 2:  # 此时文件夹内只有刚刚复制过来的初次检查图片和car_info.txt
                            print(f"{t0_dir}开始下载图片。。。")
                            get_check_imgs(stream_prefix, img_dir_full)
                        else:
                            for img_name in img_list:
                                if img_name[-4:] == '.txt':
                                    print("跳过.txt文件。")
                                    continue
                                # 二次检查
                                # 构建测试图片完整路径
                                images = [double_check_dir + ip_dir + '/' + t0_dir + '/' + img_name]
                                # 预测并将结果保存到当前目录的 output文件夹中
                                trainer.predict(images, draw_threshold=0.65, output_dir=output_path, save_txt=True)
                                # 读取txt文件信息，每一行为一个列表，包含6个元素
                                output_file = output_path + img_name[:-4] + '.txt'
                                categories, confident_val, bbox = read_txt(output_file)
                                # 二次检查的人车位置判断
                                check_res = get_check_status(roi_HongDa_in_2nd, categories, bbox)
                                print('check_res = ', check_res)
                                if check_res != '2':
                                    print('Check status remains the same.')
                                    continue
                                else:
                                    # 从txt文件读取车牌和车辆类型信息
                                    with open(double_check_dir + ip_dir + '/' + t0_dir + '/car_info.txt', 'r') as f:
                                        car_info = f.read().split('_')

                                    # 生成新的检查图片
                                    img_check = cv2.imread(images[0])
                                    img_old = cv2.imread(double_check_dir+ip_dir+'/'+t0_dir+'/' +
                                                         ip_dir+'_01_'+t0_dir+'_VEHICLE_DETECTION.jpg')
                                    h, w, chanel = img_old.shape
                                    img_check = cv2.resize(img_check, (w, h), interpolation=cv2.INTER_AREA)

                                    # cv2.namedWindow('img check', cv2.WINDOW_NORMAL)
                                    # cv2.resizeWindow("img check", 500, 500)
                                    # cv2.imshow('img check', img_check)
                                    # cv2.waitKey()

                                    # 合成新的查车图片：上面是初查，下面是二次检查
                                    img_new = np.vstack([img_old, img_check])
                                    # cv2.namedWindow('img new', cv2.WINDOW_NORMAL)
                                    # cv2.resizeWindow("img new", 500, 500)
                                    # cv2.imshow('img new', img_new)
                                    # cv2.waitKey()
                                    # cv2.destroyAllWindows()

                                    # 保存新图片到http/ip/
                                    cv2.imwrite(pathDoneFile+ip_dir + "/" + img_name, img_new)
                                    print("新图片已经保存到： %s" % pathDoneFile+ip_dir + "/" + img_name)

                                    # 构建推送数据
                                    data = {
                                        "path": "http://211.103.164.196:9080/http/" + ip_dir + "/" + img_name,
                                        "car": car_info[0],
                                        "des": car_info[1],
                                        "check": '2',
                                        "cflag": car_info[2]
                                    }
                                    print(data)
                                    if post_trigger:
                                        post_result = requests.post(url=post_url, data=data)
                                        print("php:" + post_result.text)
                                        # if post_result.text == "success":
                                        #     shutil.rmtree(double_check_dir + ip_dir + '/' + t0_dir)
                                        #     print('php succeed.')
                                    break
                            # 当遍历完t0文件夹后，如果check_res == '1'，那么发送数据
                            if check_res != '2':
                                # 从txt文件读取车牌和车辆类型信息
                                with open(double_check_dir + ip_dir + '/' + t0_dir + '/car_info.txt', 'r') as f:
                                    car_info = f.read().split('_')

                                # 构建推送数据
                                data = {
                                    "path": "http://211.103.164.196:9080/http/" + ip_dir + "/"
                                            + ip_dir + "_01_" + t0_dir + "_VEHICLE_DETECTION.jpg",
                                    "car": car_info[0],
                                    "des": car_info[1],
                                    "check": '1',
                                    "cflag": car_info[2]
                                }
                                print(data)
                                if post_trigger:
                                    post_result = requests.post(url=post_url, data=data)
                                    print("php:" + post_result.text)
                                if move_file_trigger:
                                    mymovefile(double_check_dir + ip_dir + '/' + t0_dir + "/"
                                               + ip_dir + "_01_" + t0_dir + "_VEHICLE_DETECTION.jpg",
                                               pathDoneFile + ip_dir + "/"
                                               + ip_dir + "_01_" + t0_dir + "_VEHICLE_DETECTION.jpg")
                            shutil.rmtree(double_check_dir + ip_dir + '/' + t0_dir)

    except:
        print("！！！异常错误：")

    print("sleep 30s")
    time.sleep(30)

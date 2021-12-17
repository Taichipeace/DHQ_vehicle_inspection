# coding=UTF-8
# ATAI 更新日志 2021.12.08 21：00
# 修正了几个Bug：
# 1. 之前是取的出口图片，现在切换到了入口图片
# 2. ROI区域根据入口摄像头校准
# 3. 修正车辆在ROI区域内的判断规则
# 4. 修正了OCR判断车辆类型为 '其他' 时的检查函数（这个就是导致，当图片不是0Kb，还出现 '异常错误' 的原因）
# 5. 增加了对车牌最后一位是 '挂' 的判断。
# 6. 关闭OCR的系统提示：PaddleOCR/tools/infer/predict_system.py， L62、L81。需要重新编译PaddleOCR
# 7. 把读取txt文件抽取为函数 read_txt(file)
#
# ATAI 更新日志 2021.12.09 21：25
# 1. 增加 make_t0_dir 函数，用于在二次查车时，在D盘新建t0时间点的文件夹
# 2. 增加是否POST到服务的开关
# 3. 增加是否移动图片的开关
#
# ATAI 更新日志 2021.12.11 08：59
# 新增处理初次检查为’应查未查‘的二次检查分支
#
# ATAI 更新日志 2021.12.17 09：00
# # 以下2行 2021.12.17日 Ted 添加，为解决truncated image问题
# # venv/Lib/site-packages/paddledet-2.3.0-py3.9.egg/ppdet/engine/trainer.py
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# ============================================================
# ================== 注意推送和移动图片的开关状态 ！！！！！==========
# ============================================================
import requests
import time
from paddleocr import PaddleOCR
import ppdet
from TedTools import *

def make_t0_dir(t0_path):
    if os.path.exists(t0_path):
        pass
    else:
        os.makedirs(t0_path)
        print("%s dir has been made." % t0_path)


def get_ocr_data(ocr_result):
    txt = []
    for content in ocr_result:
        txt.append(content[1][0])
        # print(content)
    # print(txt)

    # 把结果list转换成str
    Alltxt = ''.join(txt)
    print(Alltxt)

    # 提取车牌号码
    # 定位车牌起始位置
    if '牌' in Alltxt:
       LP_start_indx = Alltxt.find('牌') + 4
    # 定位车牌结束位置
    if '颜' in Alltxt:
        LP_end_indx = Alltxt.find('颜') - 2
    # 当有车牌时，去掉车牌颜色那个字符
    if (LP_end_indx - LP_start_indx) > 3:
        LP_start_indx = LP_start_indx + 1

    # 提取车辆类型
    if Alltxt.find('类型：') >= 0:
        Type_start_indx = Alltxt.find('类型：') + 3
    else:
        Type_start_indx = Alltxt.find('类型') + 2
    Type_end_indx = Alltxt.find('品牌')

    # 构建输出list
    car_licence_type = list()
    car_licence_type.append(Alltxt[LP_start_indx:LP_end_indx])
    car_licence_type.append(Alltxt[Type_start_indx:Type_end_indx])

    # 矫正识别错字产生的偏差
    if '软' in car_licence_type[1]:
        car_licence_type[1] = '轿车'
    elif '轿' in car_licence_type[1]:
        car_licence_type[1] = '轿车'
    elif 'MP' in car_licence_type[1]:
        car_licence_type[1] = 'SUV/MPV'
    elif '大' in car_licence_type[1]:
        car_licence_type[1] = '大货车'
    elif '小' in car_licence_type[1]:
        car_licence_type[1] = '小货车'
    elif '货' in car_licence_type[1]:
        car_licence_type[1] = '货车'
    elif '面' in car_licence_type[1]:
        car_licence_type[1] = '面包车'
    elif '客' in car_licence_type[1]:
        car_licence_type[1] = '客车'
    elif '皮卡' in car_licence_type[1]:
        car_licence_type[1] = '皮卡'
    else:
        car_licence_type[1] = '其他'
    # 提取的车牌号超过8位或不足6位，则有异常，按‘无车牌’输出
    if len(car_licence_type[0]) > 8 or len(car_licence_type[0]) < 6:
        car_licence_type[0] = '无车牌'
    # 如果车辆类型位数超过10位或空，则按‘无’类型输出
    if len(car_licence_type[1]) > 10 or len(car_licence_type[1]) < 1:
        car_licence_type[1] = '其他'

    # 全部转大写字母
    car_licence_type[0] = car_licence_type[0].upper()

    return car_licence_type


# 是否推送数据到服务器的开关
post_trigger = False

# 是否移动图片的开关
move_file_trigger = True

# POST地址
url = "http://daxing.smart-ideas.com.cn/inputcarocr.php"

# 初始化OCR模型
ocr = PaddleOCR(det_model_dir='TedMod/ch_PP-OCRv2_det_infer',
                rec_model_dir='TedMod/ch_PP-OCRv2_rec_infer',
                cls_model_dir='TedMod/ch_ppocr_mobile_v2.0_cls_infer',
                use_mp=True,
                total_process_num=6,
                use_angle_cls=False,
                use_gpu=False,
                det=False,
                det_limit_side_len=1920)

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

# 无需检查的车辆类型
check_pass_emu = ['轿车', '客车', 'SUV/MPV', '皮卡']

# 需要检查的车辆类型
check_must_emu = ['大货车', '小货车', '货车', '面包车']

# 处理图片
ftp_dir = "D:/workpath/ftp/"
http_dir = "D:/workpath/http/"
double_check_dir = "D:/workpath/ftp_double_check/"
while 1:
    ip_dir_list = os.listdir(ftp_dir)
    for ip_dir in ip_dir_list:
        print(f'================= Begin to process {ftp_dir}{ip_dir} ============================')
        ip_dir_fullpath = ftp_dir + ip_dir
        # 读取保存到pkl文件的pts列表
        roi_contour = read_roi_contour(f'TedRoiFiles/PolyRoi_{ip_dir}_1st.pkl')
        print(f'ROI points loaded successfully: TedRoiFiles/PolyRoi_{ip_dir}_1st.pkl')
        if os.path.isdir(ip_dir_fullpath):
            img_name_list = os.listdir(ip_dir_fullpath)
            # print("dir:"+ip_dir_fullpath)
            for img_name in img_name_list:
                if not (img_name.endswith("DETECTION.jpg")):
                    print("not pic")
                    if move_file_trigger:
                        mymovefile(img_fullpath, done_fullpath)
                    continue
                try:
                    print("================= Processing new image ")
                    img_fullpath = ip_dir_fullpath + "/" + img_name
                    print("Current image: " + img_fullpath)
                    done_fullpath = http_dir + ip_dir + "/" + img_name
                    img = cv2.imread(img_fullpath)
                    if img is not None:
                        img_cropped = img[1060:, :]
                        img_cropped = cv2.copyMakeBorder(img_cropped, 10, 10, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                        # OCR识别
                        ocr_result = ocr.ocr(img_cropped, cls=False)

                        # 处理OCR结果
                        car_info = get_ocr_data(ocr_result)
                        # print(car_info)

                        # 预测并将结果保存到当前目录的 {output_path} 文件夹中
                        trainer.predict([img_fullpath], draw_threshold=0.5, output_dir=output_path, save_txt=True)

                        # 读取txt文件信息，每一行为一个列表，包含6个元素
                        output_file = output_path + img_name[:-4] + '.txt'
                        categories = []
                        categories, confident_val, bbox = read_txt(output_file)

                        # 判断ROI区域内是否有车
                        vehicle_center = find_any_vehicle(roi_contour, categories, bbox, ['car', 'bus', 'truck'])
                        if np.count_nonzero(vehicle_center) == 0:
                            count_flag = '1'
                        else:
                            count_flag = '2'

                    else:
                        print('！！！Img is 0 Kb!')
                        os.remove(img_fullpath)
                        continue

                    # 如果图片来自摄像头192.168.10.11（宏大入口），则进行查车状态初步判断
                    if '192.168.10.11' in img_name:
                        # 提取ocr识别的车牌和车辆类型
                        car_LP = car_info[0]
                        car_type = car_info[1]

                        # 判断检查车辆的状态
                        if car_type in check_pass_emu and car_LP[-1] != '挂':
                            check_res = '0'
                        elif car_type in check_must_emu or car_LP[-1] == '挂':
                            check_res = get_check_status(roi_contour, categories, bbox)
                        else:  # 如果ocr识别的车辆类型是 ’其他‘，那么需要进一步根据det识别的车辆类型判断
                            if 'bus' in categories or 'truck' in categories:
                                check_res = get_check_status(roi_contour, categories, bbox)
                            else:
                                check_res = '0'
                        print(check_res)
                    else:
                        check_res = '0'
                        print('图片不是来自 宏大入口，无需检查车辆')

                    # 如果检查结果是’应查未查‘，则以当前时间点t0 创建新的目录 D:/workpath/ftp_double_check/192.168.x.x/t0/
                    if check_res == '1':
                        t0 = img_name.split('_')[-3]
                        t0_dir = double_check_dir+ip_dir+'/' + t0
                        make_t0_dir(t0_dir)
                        mymovefile(img_fullpath, t0_dir + "/" + img_name)
                        with open(t0_dir + "/" + 'car_info.txt', 'w') as f:
                            # f.write(car_info[0]+'_'+car_info[1])
                            f.write('_'.join(car_info) + '_' + count_flag)
                        print('初检：应查未查，不推送数据，图片已经移动到t0文件夹。')
                    else:
                        # 构建推送数据 data,
                        data = {
                            "path": "http://211.103.164.196:9080/http/" + ip_dir + "/" + img_name,
                            "car": car_info[0],
                            "des": car_info[1],
                            "check": check_res,
                            "cflag": count_flag
                        }
                        print(data)

                        # 推送初次检查数据到服务器
                        if post_trigger:
                            res = requests.post(url=url, data=data)
                            print("php:" + res.text)

                        # 推送成功后将图片移动到D:/workpath/http对应的目录
                        if move_file_trigger:
                            if res.text == "success":
                                mymovefile(img_fullpath, done_fullpath)
                except:
                    print("！！！文件异常，已跳过：" + img_fullpath)
                    if move_file_trigger:
                        mymovefile(img_fullpath, done_fullpath)
    print("sleep 30s")
    time.sleep(30)

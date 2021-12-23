from paddleocr import PaddleOCR, draw_ocr
from TedTools import *


def get_ocr_data_new(ocr_result):
    car_licence_type = ['', '']
    txt = []
    for content in ocr_result:
        if '无' in content[1][0]:
            car_licence_type[0] = '无车牌'
        elif '牌号' in content[1][0]:
            car_licence_type[0] = content[1][0][6:]

        # if '类型' in content[1][0]:
        #     car_licence_type[1] = content[1][0][content[1][0].find('类型') + 3:]

        txt.append(content[1][0])

    # 把结果list转换成str
    Alltxt = ''.join(txt)
    print(Alltxt)

    # 提取车辆类型
    if Alltxt.find('类型：') >= 0:
        Type_start_indx = Alltxt.find('类型：') + 3
    else:
        Type_start_indx = Alltxt.find('类型') + 2
    Type_end_indx = Alltxt.find('品牌')
    car_licence_type[1] = Alltxt[Type_start_indx:Type_end_indx]

    # 规范化车辆类型的输出
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

    # 矫正识别错字产生的偏差
    car_licence_type[0] = car_licence_type[0].replace('"', 'Y')

    # # 提取的车牌号超过8位或不足6位，则有异常，按‘无车牌’输出
    # if len(car_licence_type[0]) > 8 or len(car_licence_type[0]) < 6:
    #     car_licence_type[0] = '车牌待查'

    if '颜' in car_licence_type[0]:
        car_licence_type[0] = car_licence_type[0][:car_licence_type[0].find('颜') - 2]
    # 如果车辆类型位数超过10位或空，则按‘其他’类型输出
    if len(car_licence_type[1]) > 10 or len(car_licence_type[1]) < 1:
        car_licence_type[1] = '其他'

    # 全部转大写字母
    car_licence_type[0] = car_licence_type[0].upper()

    return car_licence_type


img_path = 'D:/MVproject/ceshi/00TedOCR/TestImg/192.168.10.11_01_20211223100318344_VEHICLE_DETECTION.jpg'

# 初始化OCR模型
ocr = PaddleOCR(det_model_dir='TedMod/ch_PP-OCRv2_det_infer/',
                rec_model_dir='TedMod/ch_PP-OCRv2_rec_infer/',
                cls_model_dir='TedMod/ch_ppocr_mobile_v2.0_cls_infer/',
                use_mp=True,
                total_process_num=6,
                use_angle_cls=False,
                use_gpu=False,
                det=False,
                det_limit_side_len=1920)

# 处理图片
img = cv2.imread(img_path)
img_cropped = img[1060:, :]
img_cropped = cv2.copyMakeBorder(img_cropped, 10, 10, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# OCR识别
ocr_result = ocr.ocr(img_cropped, cls=False)

# 处理OCR结果
car_info = get_ocr_data_new(ocr_result)
print(car_info)

# # 显示结果
# from PIL import Image
#
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in ocr_result]
# txts = [line[1][0] for line in ocr_result]
# scores = [line[1][1] for line in ocr_result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')

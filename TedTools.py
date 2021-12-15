import os
import shutil
import numpy as np
import cv2
import joblib


def read_roi_contour(camera_name):
    pkl_file_name = 'TedRoiFiles/PolyRoi_' + camera_name + '.pkl'
    dict_loaded = joblib.load(pkl_file_name)
    pts_loaded = dict_loaded['ROI']
    print("%s ROI points loaded successfully." % pkl_file_name)
    # print(pts_loaded)
    roi_contour = np.array(pts_loaded, dtype=np.int32)
    return roi_contour


def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))


def find_vehicle(roi_cntr, cates, bboxs):
    veh_center = np.array([0, 0])
    for cat_id, cat_name in enumerate(cates):
        if cat_name == 'bus' or cat_name == 'truck':
            in_roi = cv2.pointPolygonTest(roi_cntr, (bboxs[cat_id][0], bboxs[cat_id][1]+bboxs[cat_id][3]), False)
            if in_roi == 1 or in_roi == 0:
                veh_center = [bboxs[cat_id][0]+bboxs[cat_id][2]/2, bboxs[cat_id][1]+bboxs[cat_id][3]/2]
                break
    return veh_center


def calcu_person_distances(veh_center, cates, bboxs):
    distances = []
    for cat_id, cates_name in enumerate(cates):
        if cates_name == 'person':
            pers_center = [bboxs[cat_id][0]+bboxs[cat_id][2]/2, bboxs[cat_id][1]+bboxs[cat_id][3]/2]
            distances.append(np.sqrt(np.sum(np.square(np.array(veh_center) - np.array(pers_center)))))
    return distances


def get_check_status(roi_cntr, cates, bboxs):
    veh_center = find_vehicle(roi_cntr, cates, bboxs)
    if veh_center[0] == 0 and veh_center[1] == 0:
        result = '0'
    else:
        if 'person' not in cates:
            result = '1'
        else:
            distance_threshold = 400
            distances = calcu_person_distances(veh_center, cates, bboxs)
            if min(distances) > distance_threshold:
                result = '1'
            else:
                result = '2'
    return result


def read_txt(file):
    cates = []
    confi_val = []
    blob_box = []
    with open(file, 'r') as f:
        for line in f:
            datas = list(line.strip('\n').split(' '))
            cates.append(datas[0])
            confi_val.append(float(datas[1]))
            blob_box.append(list(map(float, datas[2:])))
    print('Data read from txt. \n')
    return cates, confi_val, blob_box

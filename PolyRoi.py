# 选取多边形ROI
#     多边形ROI，主要利用鼠标交互进行绘制：
#
#    1. 单击左键，选择多边形的点；
#
#    2. 单击右键，删除最近一次选择的点；
#
#    3. 单击中键，确定ROI区域并可视化。
#
#    4. 按”S“键，将多边形ROI区域的点保存到本地”config.pkl"文件中。
# ————————————————

import cv2
import imutils
import numpy as np
import joblib

pts = []  # 用于存放点


# 统一的：mouse callback function
def draw_roi(event, x, y, flags, param):
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，选择点
        # pts.append((x, y))
        pts.append([x, y])  # Ted modified

    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，取消最近一次选择的点
        pts.pop()

    if event == cv2.EVENT_MBUTTONDOWN:  # 中键绘制轮廓
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        # 画多边形
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 用于 显示在桌面的图像

        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    if len(pts) > 0:
        # 将pts中的最后一点画出来
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        # 画线
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 为鼠标点击地方的坐标
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)


# 创建图像与窗口并将窗口与回调函数绑定
img = cv2.imread("D:/workpath/ftp/192.168.1.10/192.168.1.10_01_20211216102039982_VEHICLE_DETECTION.jpg")
h, w, c = img.shape
img = imutils.resize(img, height=int(h/1), width=int(w/1))
cv2.namedWindow('image', 0)
cv2.setMouseCallback('image', draw_roi)
print("[INFO] 单击左键：选择点，单击右键：删除上一次选择的点，单击中键：确定ROI区域")
print("[INFO] 按‘S’确定选择区域并保存")
print("[INFO] 按 ESC 退出")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        # # 在pts列表中加入第一个点，保证区域闭合（需要测试是否有必要这么做：（实测没必要））
        # pts.append(pts[0])

        saved_data = {
            "ROI": pts
        }
        joblib.dump(value=saved_data, filename="PolyRoi.pkl")
        print("[INFO] ROI坐标已保存到本地.")
        break
cv2.destroyAllWindows()

# 读取保存到pkl文件的pts列表
dict_loaded = joblib.load('PolyRoi.pkl')
pts_loaded = dict_loaded['ROI']
print(pts_loaded)

# 判断某个点是否在ROI内部
point_test = [600, 600]
roi_contour = np.array(pts_loaded, dtype=np.int32)
test_result = cv2.pointPolygonTest(roi_contour, point_test, False)
print(test_result)

# 可视化验证
cv2.drawContours(img, [roi_contour], -1, (0, 255, 0), 3)
cv2.circle(img, point_test, 5, (0, 0, 255), 2)

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', int(w/1), int(h/1))
cv2.imshow('test', img)
cv2.waitKey()
cv2.destroyAllWindows()

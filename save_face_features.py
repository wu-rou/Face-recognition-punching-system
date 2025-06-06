#接收到ttyUSB1的信号，进行人脸检测并提取特征，保存到facebank文件夹中，在终端中给人脸和特征命名
import cv2
import numpy as np
import os
import logging
import time
import serial
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def adjust_brightness(image, factor=5.5):
    """
    提高整个图像亮度
    :param image: 输入BGR图像
    :param factor: 亮度增强因子
    :return: 增强后的图像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] *= factor
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return enhanced

def bgr2nv12_opencv(image):
    height, width = image.shape[:2]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
    nv12 = np.zeros_like(yuv420p)
    nv12[:area] = y
    nv12[area:] = uv_packed
    return nv12

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(1, (box1[2] - box1[0])) * max(1, (box1[3] - box1[1]))
    box2_area = max(1, (box2[2] - box2[0])) * max(1, (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms(boxes, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        curr = boxes.pop(0)
        keep.append(curr)
        boxes = [box for box in boxes if compute_iou(curr, box) < iou_threshold]
    return keep

def postprocess_face3(scores_tensor, boxes_tensor, origin_image, input_size=(640, 480), conf_threshold=0.6):
    ori_h, ori_w = origin_image.shape[:2]
    input_w, input_h = input_size

    scores = np.array(scores_tensor.buffer, dtype=np.float32).reshape((1, 17640, 2))[0]
    boxes = np.array(boxes_tensor.buffer, dtype=np.float32).reshape((1, 17640, 4))[0]

    results = []
    for box, score_pair in zip(boxes, scores):
        score = score_pair[1]
        if score < conf_threshold:
            continue
        x1 = int(max(0, min(box[0] * ori_w, ori_w - 1)))
        y1 = int(max(0, min(box[1] * ori_h, ori_h - 1)))
        x2 = int(max(0, min(box[2] * ori_w, ori_w - 1)))
        y2 = int(max(0, min(box[3] * ori_h, ori_h - 1)))
        results.append([x1, y1, x2, y2, float(score)])

    results = nms(results, iou_threshold=0.4)
    return results

def extract_feature(face_img, model):
    resized = cv2.resize(face_img, (112, 112))
    nv12 = bgr2nv12_opencv(resized).astype(np.uint8)
    output = model.forward(nv12)[0].buffer
    return np.array(output).flatten()

def save_face_and_feature(face_img, feature, facebank_path="facebank"):
    if not os.path.exists(facebank_path):
        os.makedirs(facebank_path)

    cv2.imshow("Please enter name in terminal then press any key to continue", face_img)
    cv2.waitKey(1)

    name = input("请输入该人脸的姓名（作为文件名保存）: ").strip()
    if name == "":
        logging.warning("名字为空，跳过保存")
        return False

    img_path = os.path.join(facebank_path, f"{name}.jpg")
    npy_path = os.path.join(facebank_path, f"{name}.npy")

    cv2.imwrite(img_path, face_img)
    np.save(npy_path, feature)
    logging.info(f"已保存人脸图像和特征: {img_path}, {npy_path}")
    return True

def main():
    detect_model = dnn.load('./models/version-RFB-640.bin')[0]
    ext_model = dnn.load('./models/w600k_r50.bin')[0]

    cam_x3pi = srcampy.Camera()
    cam_x3pi.open_cam(0, 1, 30, 1920, 1080)
    logging.info("MIPI摄像头已启动")

    try:
        ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=0)
        logging.info("串口 /dev/ttyUSB1 打开成功")
    except Exception as e:
        logging.error(f"打开串口失败: {e}")
        return

    last_trigger_time = 0

    while True:
        origin_image_x3pi = cam_x3pi.get_img(2, 1920, 1080)
        if origin_image_x3pi is None:
            logging.warning("获取图像失败，跳过")
            continue

        nv12_x3pi = np.frombuffer(origin_image_x3pi, dtype=np.uint8).reshape(1080 * 3 // 2, 1920)
        frame = cv2.cvtColor(nv12_x3pi, cv2.COLOR_YUV2BGR_NV12)

        # ⭐ 整体提亮画面
        frame = adjust_brightness(frame, factor=1.8)
        display_frame = frame.copy()

        data = ser.read(100)
        now = time.time()

        if data and (now - last_trigger_time > 2):
            last_trigger_time = now
            logging.info("串口信号触发特征提取")

            resized = cv2.resize(frame, (640, 480))
            nv12_data = bgr2nv12_opencv(resized)
            try:
                outputs = detect_model.forward(nv12_data)
                boxes = postprocess_face3(outputs[0], outputs[1], origin_image=frame, input_size=(640, 480))
            except Exception as e:
                logging.error(f"人脸检测失败: {e}")
                continue

            if not boxes:
                logging.info("没有检测到人脸")
                continue

            for (x1, y1, x2, y2, score) in boxes:
                face_img = frame[y1:y2, x1:x2]
                try:
                    feature = extract_feature(face_img, ext_model)
                    saved = save_face_and_feature(face_img, feature)
                    if not saved:
                        logging.info("保存失败或被跳过")
                except Exception as e:
                    logging.error(f"特征提取或保存失败: {e}")

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

        small_display = cv2.resize(display_frame, (960, 540))
        cv2.imshow("MIPI Camera", small_display)

    cam_x3pi.close_cam()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

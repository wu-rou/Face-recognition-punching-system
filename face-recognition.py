#接收到ttyUSB1的信号，进行人脸检测并提取特征，然后与facebank中的特征进行匹配，匹配成功即识别成功
import cv2
import numpy as np
import os
import logging
import time
import serial
import requests
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def load_facebank(facebank_path="facebank"):
    face_dict = {}
    for file in os.listdir(facebank_path):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            feature = np.load(os.path.join(facebank_path, file)).flatten()
            face_dict[name] = feature
    return face_dict

def cosine_similarity(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def send_to_server(username, recognition_time):
    api_url = "http://xxx.xxx.xxx.xxx/api/recognition-data/"
    payload = {"user_id": username, "recognition_time": recognition_time}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=5)
        if response.status_code == 201:
            logging.info("签到日志已保存")
        else:
            logging.warning(f"签到日志保存失败: {response.status_code}")
    except requests.exceptions.Timeout:
        logging.error("网络请求超时")
    except Exception as e:
        logging.error(f"发送签到日志异常: {e}")

def main():
    detect_model = dnn.load('./models/version-RFB-640.bin')[0]
    ext_model = dnn.load('./models/w600k_r50.bin')[0]
    facebank = load_facebank()
    logging.info(f"加载人脸库成功，共 {len(facebank)} 人")

    cam_x3pi = srcampy.Camera()
    cam_x3pi.open_cam(0, 1, 30, 1920, 1080)
    logging.info("MIPI摄像头已启动")

    try:
        ser = serial.Serial('/dev/ttyUSB1', 9600, timeout=0.1)
        logging.info("串口 /dev/ttyUSB1 打开成功")
    except Exception as e:
        logging.error(f"串口 /dev/ttyUSB1 打开失败: {e}")
        return

    try:
        ser_out = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.1)
        logging.info("串口 /dev/ttyUSB0 打开成功")
    except Exception as e:
        logging.error(f"串口 /dev/ttyUSB0 打开失败: {e}")
        ser_out = None

    last_trigger_time = 0

    while True:
        origin_image_x3pi = cam_x3pi.get_img(2, 1920, 1080)
        if origin_image_x3pi is None:
            logging.warning("获取图像失败，跳过")
            continue

        nv12_x3pi = np.frombuffer(origin_image_x3pi, dtype=np.uint8).reshape(1080 * 3 // 2, 1920)
        frame = cv2.cvtColor(nv12_x3pi, cv2.COLOR_YUV2BGR_NV12)

        # 增加画面亮度，alpha是增益，beta是亮度偏移
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

        try:
            serial_data = ser.read(100)
        except Exception as e:
            logging.error(f"串口读取错误: {e}")
            serial_data = b''

        current_time = time.time()
        if serial_data and (current_time - last_trigger_time >= 2):
            logging.info("收到串口数据，开始识别")
            last_trigger_time = current_time

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
                # 画出人脸检测框和分数
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                face_img = frame[y1:y2, x1:x2]
                try:
                    feature = extract_feature(face_img, ext_model)

                    best_name = "未知"
                    best_score = 0.0
                    for name, ref_feat in facebank.items():
                        sim = cosine_similarity(feature, ref_feat)
                        if sim > best_score:
                            best_score = sim
                            best_name = name

                    if best_score >= 0.5:
                        label = f"{best_name} ({best_score:.2f})"
                        recognition_time = time.strftime('%Y-%m-%d %H:%M:%S')
                        send_to_server(best_name, recognition_time)
                        logging.info(f"识别成功，姓名：{best_name}")
                        if ser_out:
                            try:
                                ser_out.write(b'<G>\xca\xb6\xb1\xf0\xb3\xc9\xb9\xa6')
                                logging.info("已向 /dev/ttyUSB0 发送成功信号")
                            except Exception as e:
                                logging.error(f"发送串口数据失败: {e}")
                    else:
                        logging.info(f"未知人脸，相似度为 {best_score:.2f}，不发送信号")

                except Exception as e:
                    logging.error(f"识别失败: {e}")

        small_display = cv2.resize(frame, (960, 540))
        cv2.imshow("MIPI Camera", small_display)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cam_x3pi.close_cam()
    ser.close()
    if ser_out:
        ser_out.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
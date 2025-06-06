#人脸检测，按下空格键对此时画面进行人脸检测，显示3秒
import cv2
import numpy as np
import logging
import time
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

def main():
    detect_model = dnn.load('./models/version-RFB-640.bin')[0]

    cam_x3pi = srcampy.Camera()
    cam_x3pi.open_cam(0, 1, 30, 1920, 1080)
    logging.info("MIPI摄像头已启动")

    try:
        while True:
            origin_image_x3pi = cam_x3pi.get_img(2, 1920, 1080)
            if origin_image_x3pi is None:
                logging.warning("获取图像失败，跳过")
                continue

            nv12_x3pi = np.frombuffer(origin_image_x3pi, dtype=np.uint8).reshape(1080 * 3 // 2, 1920)
            frame = cv2.cvtColor(nv12_x3pi, cv2.COLOR_YUV2BGR_NV12)
            display_frame = frame.copy()

            key = cv2.waitKey(1)

            if key == 32:  # 空格键
                logging.info("空格键触发人脸检测并定格画面")
                freeze_frame = display_frame.copy()
                resized = cv2.resize(freeze_frame, (640, 480))
                nv12_data = bgr2nv12_opencv(resized)

                try:
                    outputs = detect_model.forward(nv12_data)
                    boxes = postprocess_face3(outputs[0], outputs[1], origin_image=freeze_frame, input_size=(640, 480))
                except Exception as e:
                    logging.error(f"人脸检测失败: {e}")
                    boxes = []

                if boxes:
                    for (x1, y1, x2, y2, score) in boxes:
                        cv2.rectangle(freeze_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(freeze_frame, f"{score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    logging.info(f"检测到 {len(boxes)} 张人脸")
                else:
                    logging.info("未检测到人脸")

                # 冻结画面显示 3 秒
                frozen_display = cv2.resize(freeze_frame, (960, 540))
                cv2.imshow("MIPI Camera - Face Detection (Frozen)", frozen_display)
                cv2.waitKey(1)
                time.sleep(3)

            elif key == 27 or key == ord('q'):  # ESC 或 q 键退出
                break

            small_display = cv2.resize(display_frame, (960, 540))
            cv2.imshow("MIPI Camera - Press Space to Detect Face", small_display)

    finally:
        cam_x3pi.close_cam()
        cv2.destroyAllWindows()
        logging.info("摄像头关闭，程序退出")

if __name__ == '__main__':
    main()
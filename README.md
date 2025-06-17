# Face-recognition-punching-system
能在地平线旭日x3上本地运行的人脸识别打卡功能


face-e.py是可以直接运行的特征提取到人脸库的代码
face-r.py是可以直接进行人脸识别的代码


1.save_face_features.py

监听 /dev/ttyUSB1 串口信号触发人脸检测、特征提取，同时在终端中输入名称，人脸图和特征npy将会被命名然后保存到facebank文件夹中。
触发时机：当串口收到信号时，启动摄像头抓取一帧图像，进行人脸检测。
检测到人脸：从图像中裁剪出人脸区域，使用人脸特征提取模型提取128维特征向量。
保存结果：弹出窗口展示人脸，终端要求输入姓名，保存人脸图像和特征向量到 facebank 文件夹。



2.face-recognition.py

监听 /dev/ttyUSB1 串口信号触发人脸检测、特征提取，和facebank 中的特征做余弦相似度匹配，匹配成功则识别成功，匹配成功后发送HTTP签到日志用来上传用户名和时间并向/dev/ttyUSB0 串发送识别成功信号。


3.face-e.py

和save_face_features.py功能一样，只是不需要串口连接，按空格触发。
4.face-r.py

和save_face_features.py功能一样，不需要串口连接，按空格触发，没有语音播报。

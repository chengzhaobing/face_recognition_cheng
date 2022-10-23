import cv2
import os


def start_capture(name):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
    try:
        os.makedirs(path)
    except:
        print('文件夹创建完成')
        # vid = cv2.VideoCapture(0)  # 外接摄像头1，2
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, img = vid.read()  # 获取摄像头拍摄画面
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        # 1. image为输入的灰度图像 scaleFactor为每一个图像尺度中的尺度参数，默认值为1.1。2.scale_factor参数可以决定两个不同大小的窗口扫描之间有多大的跳跃，这个参数设置的大，则意味着计算会变快，但如果窗口错过了某个大小的人脸，则可能丢失物体。3.minNeighbors参数为每一个级联矩形应该保留的邻近个数（没能理解这个参数，-_-|||），默认为3。minNeighbors控制着误检测，默认值为3表明至少有3次重叠检测，我们才认为人脸确实存。
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images) + " images captured"), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255))
            new_img = img[y:y + h, x:x + w]
        cv2.imshow("loadding", img)
        key = cv2.waitKey(1) & 0xFF  # 按位与  只取cv2.waitKey(1)返回值最后八位，因为有些系统cv2.waitKey(1)的返回值不止八位

        try:
            cv2.imwrite(str(path + "/" + str(num_of_images) + name + ".jpg"), new_img)
            num_of_images += 1
        except:

            pass  # q键退出  27为ESC的ASCII码   num_of_images 初始化为0 采集到310
        if key == ord("q") or key == 27 or num_of_images > 1500:
            break
    cv2.destroyAllWindows()
    return num_of_images

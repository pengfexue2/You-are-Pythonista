import cv2
import math
import dlib
from PIL import Image, ImageDraw, ImageFont

import numpy as np

def showChat():

    chatpng = "chat.png"
    #opencv启用摄像头
    cap = cv2.VideoCapture(0)
    #dlib面部识别模块相关
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #定义图片名称，temp为摄像头最初抓取的图像，result为最终处理后生成图像
    temp = "temp.jpg"
    result = "result.png"

    while True:
        #将摄像头抓取到的结果进行赋值
        _, frame = cap.read()
        #将抓到的数据写入temp图片
        cv2.imwrite(temp,frame)
        #通过PIL重新打开图片，因为后续需要PIL贴图操作，所以要使用PIL模块打开
        im = Image.open(temp)

        #在摄像头抓取的数据中进行面部识别
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            #获取面部模式
            landmarks = predictor(gray,face)
            x1, y1 = landmarks.part(16).x, landmarks.part(16).y

            chatBox= Image.open(chatpng)
            resized1 = chatBox.resize((300, 200))
            im.paste(resized1, (int(x1), int(y1-200)), resized1)
            im.save(result)

            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype('simhei.ttf', 20)
            draw.text((int(x1+50), int(y1-150)),"五一假期我回老家啦，\n在家里陪了下父母！",fill="rgb(255,255,255)",font=font)


        #将窗口定义为可调节大小
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        #将result图片展示在窗口中
        #cv2.imshow("Frame", cv2.imread(result))
        cv2.imshow("Frame",np.array(im)[:,:,::-1])

        key = cv2.waitKey(1)
        #按ESC键退出摄像头视频
        if key==27:
            break

    #退出摄像头、关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    showChat()
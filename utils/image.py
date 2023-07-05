import datetime
import os
import time
import cv2
from PIL import ImageFont, ImageDraw, Image, ImageTk
import tkinter as tk

class ImagePro():
    def __init__(self):
        # 创建窗口
        self.window = tk.Tk()
        self.window.title("Image with Text")
        self.window.geometry("800x600")

        # 创建画布
        self.canvas = tk.Canvas(self.window, width=800, height=600)
        self.canvas.pack()

        # 设置文字内容和字体
        self.font = ImageFont.truetype('font/simsun.ttc', size=50)
        # 设置文字颜色
        self.text_color = (255, 0, 0)  # 红色
        # 设置文字位置
        self.text_position = (50, 50)

        # # 打开图片对象
        # image = Image.open('back/1.bmp')
        # self.addtext('未开始',image)

    # 图像捕获
    def camera(self):
        print('开始捕获图像，按空格键确认，ESC键退出...')
        cap = cv2.VideoCapture(1)
        width = 640
        height = 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while True:
            # 读取图片
            ret, img = cap.read()
            cv2.imshow('camera', img)
            # 保持画面持续
            key = cv2.waitKey(1)
            # 空格键保存
            if key == ord(" "):
                cv2.imwrite("./image/image.png", img)
            # Esc退出
            if key == 27:
                break
        # 关闭摄像头
        cap.release()
        cv2.destroyAllWindows()

    # 图片添加文字
    def addtext(self,text, data_img):
        # # 打开图片对象
        image = Image.fromarray(data_img)
        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        # 在图片上添加文字
        draw.text(self.text_position, text, font=self.font, fill=self.text_color)
        # 显示图片
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        # 更新窗口
        self.window.update()
        # 暂停1秒钟
        # time.sleep(0.2)

    # 保存图像文件
    def saveimg(self,folder_path, text, data_img):
        # 检查文件夹是否存在，如果不存在则创建
        folder_path = 'image/' + folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            pass
        # # 将RGB图像转换为BGR图像
        # data_img = cv2.cvtColor(data_img, cv2.COLOR_RGB2BGR)
        # 将图像数据转为图像
        img = Image.fromarray(data_img)
        # 获取当前系统时间
        current_time = datetime.datetime.now()
        # 格式化时间字符串
        time_string = current_time.strftime("%Y%m%d_%H%M%S")
        # 保存图像文件
        img.save(folder_path + '/' + text + time_string + '.bmp')
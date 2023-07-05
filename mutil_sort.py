# -- coding: utf-8 --

import termios
import sys
import threading
import os
import time
from PIL import Image

import cv2
import numpy as np
from predict import *
from device import *
from utils.command import *
from utils.camera import *
from utils.image import *
from utils.utils import *

g_bExit = False
pre_lable = "unknow"  # 上一次预测结果
first_lables = []
last_lables = []
none_lable = ""
lables = read_lables()
lable_dic = {key: value for key, value in zip([str(x) for x in range(0, len(lables))], lables)}


def work_run(data_img, model, image):
	'''图像识别'''
	global pre_lable
	# 1.图像数据预处理
	temp = np.asarray(data_img)
	temp = temp.reshape((540, 720, 3))
	# temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
	# 2.模型预测
	lable = model.modelpred(temp)
	print("预测结果：", lable)
	# 3.根据预测结果输出指令
	# 如果上一个预测结果为空传送带，且这一个预测结果不是，则输出对应指令
	try:
		print(none_lable,first_lables,last_lables)
		if pre_lable == lable_dic[none_lable]:
			for i in range(len(first_lables)):
				if lable == lable_dic[first_lables[i]]:
					send_command(Command.sort_one)
			for i in range(len(last_lables)):
				if lable == lable_dic[last_lables[i]]:
					send_command(Command.sort_two)
	except:
		pass
	pre_lable = lable
	# 图像显示
	image.addtext(lable, temp)


def work_thread(camera, model):
    '''为线程定义一个函数'''
    # 0.初始化
    camera.init_MV()
    send_command(Command.init)
    image = ImagePro()
    temp = np.asarray(camera.data_buf)
    temp = temp.reshape((540, 720, 3))
    image.addtext('未开始', temp)
    while True:
        # 1.图像数据
        data_img = camera.data_buf
        # 2.识别图像
        work_run(data_img, model, image)
        # 3.更新图像
        camera.updata()
        time.sleep(0.2)
        if g_bExit == True:
            break


def press_any_key_exit():
    '''输入任意值则退出程序'''
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd)
    new_ttyinfo = old_ttyinfo[:]
    new_ttyinfo[3] &= ~termios.ICANON
    new_ttyinfo[3] &= ~termios.ECHO
    # sys.stdout.write(msg)
    # sys.stdout.flush()
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    try:
        os.read(fd, 7)
    except:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)


if __name__ == "__main__":
    # global first_lables
    # global last_lables

    # 1.选择相机
    camera = Camera()
    camera.Init()
    # 输入相机编号
    while 1:
        camera.nConnectionNum = input("please input the number of the device to connect:")

        input_result = is_int(camera.nConnectionNum)
        if not input_result:
            continue
        elif int(camera.nConnectionNum) >= camera.deviceList.nDeviceNum:
            print("intput error!")
        else:
            break

    # 2.分拣设备初始化，选择分拣位的分拣标签
    Init()
    print(lable_dic)

    first_lables = input_data("请输入第一个分拣位要分拣的标签数量：", "请输入第一个分拣位要分拣的标签索引-", lable_dic)
    last_lables = input_data("请输入第二个分拣位要分拣的标签数量：", "请输入第二个分拣位要分拣的标签索引-", lable_dic)
    while True:
        none_lable = input("请输入皮带上没有物体时的标签索引：")
        if input_lable(none_lable, lable_dic):
            break

    # 3.加载模型
    model = Model()
    model.modelinit()

    # 4.启动设备和相机
    cam, data_buf, nPayloadSize = camera.run()
    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(camera, model))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    # 5.关闭设备和相机
    print("press a key to stop grabbing.")
    press_any_key_exit()
    g_bExit = True
    hThreadHandle.join()
    send_command(Command.close_all)
    camera.close()
    time.sleep(5)
    send_command(Command.close_all)

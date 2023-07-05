# -- coding: utf-8 --

import termios
import sys
import threading
import os
import cv2
import numpy as np
from ctypes import *
from predict import predImage,read_lables,addtext
from device import *
sys.path.append("MvImport")
from MvCameraControl_class import *

g_bExit = False
first_lable = ""
last_lable = ""
none_lable = ""
lable_dic={}
# 为线程定义一个函数
def work_thread(cam=0, pData=0, nDataSize=0):
	stFrameInfo = MV_FRAME_OUT_INFO_EX()
	memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
	pre_lable = "unknow"
	is_first = True
	while True:
		temp = np.asarray(pData)
		temp = temp.reshape((540,720,3))
		temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
		lable = predImage(temp)
		# addtext(lable,temp)
		print(lable)
		# 初始化设备
		if(is_first):
			is_first=False
			send_command(Command.init)
		# 第二步 标签改变了，其他的当然也要变
		for i in range(len(first_lable)):
			try:
				if(pre_lable==lable_dic[none_lable] and lable==lable_dic[first_lable[i]]):
					send_command(Command.sort_one)
			except:
				pass
		for i in range(len(last_lable)):
			try:
				if (pre_lable==lable_dic[none_lable] and lable == lable_dic[last_lable]):
					send_command(Command.sort_two)
			except:
				pass
		pre_lable=lable
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
		if ret == 0:
			print ("get one frame: Width[%d], Height[%d], PixelType[0x%x], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.enPixelType,stFrameInfo.nFrameNum))
		else:
			print ("no data[0x%x]" % ret)
		if g_bExit == True:
				break

def press_any_key_exit():
	fd = sys.stdin.fileno()
	old_ttyinfo = termios.tcgetattr(fd)
	new_ttyinfo = old_ttyinfo[:]
	new_ttyinfo[3] &= ~termios.ICANON
	new_ttyinfo[3] &= ~termios.ECHO
	#sys.stdout.write(msg)
	#sys.stdout.flush()
	termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
	try:
		os.read(fd, 7)
	except:
		pass
	finally:
		termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)
	
if __name__ == "__main__":

	SDKVersion = MvCamera.MV_CC_GetSDKVersion()
	print ("SDKVersion[0x%x]" % SDKVersion)

	deviceList = MV_CC_DEVICE_INFO_LIST()
	tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
	
	# ch:枚举设备 | en:Enum device
	ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
	if ret != 0:
		print ("enum devices fail! ret[0x%x]" % ret)
		sys.exit()

	if deviceList.nDeviceNum == 0:
		print ("find no device!")
		sys.exit()

	print ("Find %d devices!" % deviceList.nDeviceNum)

	for i in range(0, deviceList.nDeviceNum):
		mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
		if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
			print ("\ngige device: [%d]" % i)
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
				strModeName = strModeName + chr(per)
			print ("device model name: %s" % strModeName)

			nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
			nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
			nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
			nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
			print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
		elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
			print ("\nu3v device: [%d]" % i)
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
				if per == 0:
					break
				strModeName = strModeName + chr(per)
			print ("device model name: %s" % strModeName)

			strSerialNumber = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
				if per == 0:
					break
				strSerialNumber = strSerialNumber + chr(per)
			print ("user serial number: %s" % strSerialNumber)

	if sys.version >= '3':
		nConnectionNum = input("please input the number of the device to connect:")
	else:
		nConnectionNum = raw_input("please input the number of the device to connect:")

	if int(nConnectionNum) >= deviceList.nDeviceNum:
		print ("intput error!")
		sys.exit()
	# 分拣设备初始化，启动皮带，放料
	Init()
	lables = read_lables()
	lable_dic={key:value for key,value in zip([str(x) for x in range(0,len(lables))],lables)} #字典 键：标签索引 值：标签
	print(lable_dic)
	# 第一步 更改输入
	first_num = input("请输入第一个分拣位要分拣的标签数量：")
	print(type(first_num))
	first_lable = []
	for i in range(int(first_num)):
		lable=input("请输入第一个分拣位要分拣的标签索引：")
		first_lable.append(lable)
	for i in range(len(first_lable)):
		if(first_lable[i] not in lable_dic):
			print("输入标签错误")
			sys.exit()

	last_num = input("请输入第二个分拣位要分拣的标签数量：")
	print(type(last_num))
	last_lable = []
	for i in range(int(last_num)):
		lable = input("请输入第二个分拣位要分拣的标签索引：")
		last_lable.append(lable)
	for i in range(len(last_lable)):
		if (last_lable[i] not in lable_dic):
			print("输入标签错误")
			sys.exit()

	# none_num = input("请输入皮带上没有物体时的标签索引数量：")
	# print(type(none_num))
	# none_lable = []
	# for i in range(int(none_num)):
	# 	lable = input("请输入皮带上没有物体时的标签索引：")
	# 	none_lable.append(lable)
	# for i in range(len(none_lable)):
	# 	if (none_lable[i] not in lable_dic):
	# 		print("输入标签错误")
	# 		sys.exit()
	# last_lable=input("请输入第二个分拣位要分拣的标签索引：")
	# if (last_lable not in lable_dic):
	# 	print("输入标签错误")
	# 	sys.exit()
	none_lable = input("请输入皮带上没有物体时的标签索引：")
	if (none_lable not in lable_dic):
		print("输入标签错误")
		sys.exit()
	# ch:创建相机实例 | en:Creat Camera Object
	cam = MvCamera()
	
	# ch:选择设备并创建句柄| en:Select device and create handle
	stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

	ret = cam.MV_CC_CreateHandle(stDeviceList)
	if ret != 0:
		print ("create handle fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:打开设备 | en:Open device
	ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
	if ret != 0:
		print ("open device fail! ret[0x%x]" % ret)
		sys.exit()
	
	# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
	if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
		nPacketSize = cam.MV_CC_GetOptimalPacketSize()
		if int(nPacketSize) > 0:
			ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
			if ret != 0:
				print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
		else:
			print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

	# ch:设置触发模式为off | en:Set trigger mode as off
	ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
	if ret != 0:
		print ("set trigger mode fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:获取数据包大小 | en:Get payload size
	stParam =  MVCC_INTVALUE()
	memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
	
	ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
	if ret != 0:
		print ("get payload size fail! ret[0x%x]" % ret)
		sys.exit()
	nPayloadSize = stParam.nCurValue

	# ch:开始取流 | en:Start grab image
	ret = cam.MV_CC_StartGrabbing()
	if ret != 0:
		print ("start grabbing fail! ret[0x%x]" % ret)
		sys.exit()

	data_buf = (c_ubyte * nPayloadSize)()

	try:
		hThreadHandle = threading.Thread(target=work_thread, args=(cam, data_buf, nPayloadSize))
		hThreadHandle.start()
	except:
		print ("error: unable to start thread")
		
	print ("press a key to stop grabbing.")
	press_any_key_exit()

	g_bExit = True
	hThreadHandle.join()

	# ch:停止取流 | en:Stop grab image
	ret = cam.MV_CC_StopGrabbing()
	if ret != 0:
		print ("stop grabbing fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:关闭设备 | Close device
	ret = cam.MV_CC_CloseDevice()
	if ret != 0:
		print ("close deivce fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	# ch:销毁句柄 | Destroy handle
	ret = cam.MV_CC_DestroyHandle()
	if ret != 0:
		print ("destroy handle fail! ret[0x%x]" % ret)
		del data_buf
		sys.exit()

	del data_buf
	send_command(Command.close_all)

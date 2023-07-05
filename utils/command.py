from enum import IntEnum

class Command(IntEnum):
    '''指令'''
    sort_one_open=0		  	#分拣位1放下
    sort_one_close=1		#分拣位1抬升
    sort_two_open=2			#分拣位2放下
    sort_two_close=3		#分拣位2抬升
    init=4					#初始化（两个分拣位抬上去，启动皮带，放料）
    sort_one=5			    #分拣位1分拣（执行分拣，并放料）
    sort_two=6			    #分拣位2分拣（执行分拣，并放料）
    close_all=7         	#关闭所有设备
    unknow=8				#未知指令
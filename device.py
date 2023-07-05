# encoding:utf-8
import serial

def Init():
    '''传送带初始化'''
    global serial_port 
    serial_port= serial.Serial(
            port="/dev/ttyTHS1",
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            )

def send_command(command):
    '''发送指令'''
    print(str(int(command)))
    serial_port.write(str(int(command)).encode())

if __name__ == "__main__":
    print("程序开始运行，输入q退出程序")
    Init()
    try:
        while(True):
            command=input("请输入指令：")
            if(command=="q"):
                break
            else:
                serial_port.write(command.encode())
    except Exception as ex:
        print(ex)						
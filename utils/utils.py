import sys


def is_int(data):
    '''输入格式是否正确 q退出程序 w重新输入'''
    try:
        data = int(data)
    except:
        print('请输入数字！')
        return False
    else:
        # print(f'输入{data}',type(data))
        return True

def is_indist(dist,input):
    '''输入内容是否在字典里'''

    # is_input(input)
    if (input not in dist):
        print("输入标签错误")
        return False
    return True

def input_lable(input_data,lable_dic):
    isint = is_int(input_data)
    if not isint:
        return False
    else:
        if is_indist(lable_dic, input_data):
            return True
        else:
            return False

def input_data(inputnum,inputlable,lable_dic):
    while True:
        n = 0
        lables = []
        num = input(inputnum)
        if not is_int(num):
            continue
        else:
            break
    while n < int(num):
        lable = input(inputlable + f'第{str(n+1)}个：')
        if input_lable(lable, lable_dic):
            n = n + 1
            lables.append(lable)
        else:
            continue
    return lables

def read_lables(file='./lables.txt'):
    '''读取整个文件'''
    with open(file, encoding='UTF-8') as file_read:
        return eval(file_read.read())
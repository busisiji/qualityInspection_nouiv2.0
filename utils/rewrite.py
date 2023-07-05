import sys

def get_input(data=None):
	'''重写input函数'''
	if sys.version >= '3':
		input = input(data)
	else:
		input = raw_input(data)
	return input
# -- coding: utf-8 --

import ray
from types import FunctionType, CodeType
from types import MethodType 

def createFun(f):
	code = """
def {f}(self,*args,**kwargs):
    import ray
    result = ray.get(self.obj.{f}.remote(*args,**kwargs))
    return result"""
	code= code.format(f=f)
	foo_compile = compile(code, "<string>", "exec")
	foo_code = [i for i in foo_compile.co_consts if isinstance(i, CodeType)][0]
	func = FunctionType(foo_code, globals())
	return func

class Remote(object):
	def __init__(self,obj):
		self.obj = obj
		self.setFunc()
	def	setFunc(self):
		for i in dir(self.obj):
			func = createFun(i)
			self.__setattr__(i,MethodType(func,self))


 

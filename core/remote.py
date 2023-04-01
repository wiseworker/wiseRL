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
	#print("code",code)

	# 编译为字节代码对象
	foo_compile = compile(code, "<string>", "exec")
	#print(foo_compile)

	#遍历字节代码对象，获取到code类型的对象
	foo_code = [i for i in foo_compile.co_consts if isinstance(i, CodeType)][0]
	#print(type(foo_code))

	# 1、FunctionType创建新函数（根据compile生成的code对象生成），2、在globals当前命名空间新增key
	func = FunctionType(foo_code, globals())
	return func

class Remote(object):
    def __init__(self,obj):
        self.obj = obj
        self.setFunc()
    def setFunc(self):
	    for i in dir(self.obj):
		    func = createFun(i)
		    self.__setattr__(i,MethodType(func,self))


 

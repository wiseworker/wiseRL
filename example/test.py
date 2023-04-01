from types import FunctionType, CodeType

code_1 = """
def uuid()->str:
    import uuid
    return str(uuid.uuid1())"""

# 编译为字节代码对象
foo_compile = compile(code_1, "<string>", "exec")
print(foo_compile)

# 遍历字节代码对象，获取到code类型的对象
foo_code = [i for i in foo_compile.co_consts if isinstance(i, CodeType)][0]
print(type(foo_code))

# 1、FunctionType创建新函数（根据compile生成的code对象生成），2、在globals当前命名空间新增key
#globals()['uuid'] = FunctionType(foo_code, globals())

# 执行动态添加的代码

#func = globals()['uuid']
func =  FunctionType(foo_code, globals())
print(func())
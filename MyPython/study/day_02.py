# 函数定义
def area(width, height, default = 2):
    return width * height * default
w = 4
h = 5
print('width = ', w, 'height = ', h, 'area = ', area(w, h))
print('area: ', area(width=6, height=8))


# 不定长参数函数定义
def printinfo(arg1, *vartuple):
    """打印任何传入参数"""
    print("输出:")
    print(arg1)
    for x in vartuple:
        print(x)
    return
printinfo(2)
printinfo(2, 'weijlu', 'hello')


# lambda关键字定义匿名函数
sum = lambda arg1, arg2: arg1 + arg2
print('相加后的值：',sum(3, 5))


# return语句
def testreturn(arg1, arg2):
    total = arg1 + arg2
    return #total
print(testreturn(3, 5))


# 变量作用域：关键字global(用于访问修改外部作用域变量)、nonlocal(用于访问修改嵌套作用域)
num = 1


def fun1():
    global num # 需要用global关键字声明
    print(num)
    num = 123
    print(num)
fun1()


def outer():
    seed = 10
    def inner():
        nonlocal seed
        seed = 100
        print(seed)
    inner()
    print(seed)
outer()

if __name__ == '__main__':
    print('自身程序运行')
else:
    print('来自外来程序运行')
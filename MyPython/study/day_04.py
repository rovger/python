from study import day_02

# 模块
area = day_02.area(3, 5)
print("modul call:", area)

# 读写文件
filename = 'test.txt'
# file_w = open(filename, 'w', encoding='utf-8')
# file_w.write('weijlu在往test.txt文件中写入一行字符，\n请进去文件中检查，是否已经写入成功！...')
# file_w.close()
file_r = open(filename, 'r', encoding='utf-8')
content = file_r.read()
print(content)
file_r.close()


# 面向对象
#类对象
class Myclass:
    num = 12345

    def __init__(self, num): # 构造方法，self代表类的实例，而非类，类似于Java中的this关键字
        # self.data = []
        self.num = num

    def func(self):
        return 'hello world! ' + str(self.num)
obj = Myclass(666)
print('Myclass 类属性num为：', obj.num)
print('Myclass 类方法打印为：', obj.func())


# 继承
class people:
    name = ''
    age = 0
    __weight = 0 # 定义私有属性，在类外部无法访问

    def __init__(self, name, age, weight): # 定义构造函数
        self.name = name
        self.age = age
        self.__weight = weight

    def speak(self):
        print('%s 说： 我 %d 岁!' % (self.name, self.age))
# man = people('weijlu', 25, 75)
# print(man.speak())
# 单继承
class student(people):
    grade = ''

    def __init__(self, name, age, weight, grade):
        people.__init__(self, name, age, weight)
        self.grade = grade
    # 覆盖父类方法
    def speak(self):
        print('%s 说：我今年 %d 岁了，上 %d 年级了!' % (self.name, self.age, self.grade))
stu = student('yChen', 2, 20, 1)
print(stu.speak())

# 多继承

# 方法重写
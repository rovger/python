import sys

# 斐波拉契数列
a, b = 0, 1
while b < 10:
    print(b)
    a, b = b, a+b

# end关键字：可以用于将结果输出到同一行，或者在输出的末尾添加不同的字符
a, b = 0, 1
while b < 100:
    print(b, end=',')
    a, b = b, a+b

"""section1: 条件控制"""
age = 2 # int(input("请输入你家狗的年龄："))
print("")
if age <= 0:
    print("你是在逗我吧！")
elif age == 1:
    print("相当于14岁的人。")
elif age == 2:
    print("相当于22岁的人。")
else:
    human = 22 + (age - 2) * 5
    print("对应人类的年龄：%d" % human)

"""section2: 循环语句"""
# while循环
n = 100
sum = 0
counter = 1
while counter <= n:
    sum = sum + counter
    counter += 1
print("1 到 %d之和为：%d" % (n, sum))

# for语句：可以配合break和continue关键字使用
language = ['Java', 'C++', 'C', 'Python']
for x in language:
    if x == 'weijlu':
        print(x)
        break
    print('已经拿到：%s' % x)
else:
    print('未拿到数据')

# range()函数：遍历数字序列，可以使用内置range()函数，它会生成数列
for i in range(5, 9):
    print(i)

for lan in range(len(language)):
    print(lan, language[lan])

# pass语句：pass语句是空语句，为了保证程序完整性，一般用作占位语句

# 迭代器(基本方法：iter()、next())
list = [1, 2, 3, 4]
it = iter(list)
for x in it:
    print(x, end=',')
# 或
list = ['1', '2', '3']
it = iter(list)
while True:
    try:
        print(next(it), end=';')
    except StopIteration:
        sys.exit()
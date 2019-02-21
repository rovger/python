import sys
from collections import deque
# 数据结构

# List列表：列表可以修改，字符串和元组不可以，其数据结构与堆栈Stack很像
a = [66.25, 333, 333, 1, 1234.5]
# print(a.count(333))
a.insert(1, 888)
a.append(999)
a.sort()
'''
for x in a:
    print(x, ',)
'''
"""
it = iter(a)
while True:
    try:
        print(next(it), end=',')
    except StopIteration:
        sys.exit()
"""
# List实现堆栈结构
stack = [3, 45, 44.67]
stack.append(888)
stack.append(8)
print('取出的最后一位：', stack.pop())

# List实现队列结构
queue = deque(stack)
print(queue.popleft())


# 元组Tuple
t = 23, 56, 78, 'Hello World!'
# 或
tup = (45, 89, "biubiu...")
# print(t)
print(tup[2])


# 集合Set
basket = {'apple', 'orange', 'apple', 'pear', 'banana'}
# print(basket)
print('orange' in basket)


# 字典Dictionary
tel = {'jack': 8890, 'sape': 3456}
tel['guo'] = 5555
print(list(tel.keys()))
print(list(tel.values()))
print('weijlu' in tel)
# 遍历技巧
for k, v in tel.items():
    print(k, v)
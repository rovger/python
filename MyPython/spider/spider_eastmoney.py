"""
Python爬虫
东方财富所有A股票(包括沪市，深市)：http://quote.eastmoney.com/stocklist.html
1. 获取所有{股票代码：股票链接}的mapping关系
2. 遍历mapping关系，然后获各个股票的详细数据
3. 存储CSV格式文件

坑1：
requests的content和text属性的区别：
text 返回的是unicode 型的数据，一般是在网页的header中定义的编码形式。content返回的是bytes，二进制的数据。如果想要提取文本就用text，如果想要提取图片、文件，就要用到content。

坑2：
中文乱码问题：
编码格式，要按照原来html中header定义的一致
src = requests.get(self.Url)
src.encoding = 'gb2312'
"""

from bs4 import BeautifulSoup
import requests, time, os, random, xlwt


class East(object):
    def __init__(self):
        self.Url = 'http://quote.eastmoney.com/stocklist.html'
        self.Stocks = []
        self.Date = time.strftime('%Y%m%d')
        self.fileName = self.Date + '.xls'
        # print(self.fileName)
        print('spider starting ' + time.strftime('%Y/%m/%d  %I:%M:%S'))
        if os.path.exists(self.fileName):
            print('fileName exist...')
        else:
            print('get data...')
            self.get_data()

    def get_data(self):
        # 随机延迟，方式被封ip
        time.sleep(random.randint(1, 5) + random.random())
        src = requests.get(self.Url)
        src.encoding = 'gb2312'
        soup = BeautifulSoup(src.text, 'html.parser')
        a_tags = soup.find('div', {'class': 'quotebody'}).find_all('a', {'target': '_blank'})
        counter = 0
        for a in a_tags:
            if counter == 100:
                break
            stock_data = {}
            a_text = a.get_text().split('(')
            num = a_text[1].strip(')')
            # 沪市A股以60开头，深市A股以00开头
            if not (num.startswith('00') or num.startswith('60')):
                continue
            link = a['href']
            stock_data['代码'] = num
            stock_data['名称'] = a_text[0]
            stock_data['链接'] = link

            # 根据url获取股票"公司核心数据"
            try:
                core = requests.get(link, timeout=10)
                core.encoding = 'gb2312'
            except Exception as e:
                print('call ' + a_text[0] + ' url timeout', e)
                continue
            coresoup = BeautifulSoup(core.text, 'html.parser')
            try:
                # 找到目标数据所在选择器下
                td_tags = coresoup.select('div.pad5 table.line23 td')
            except Exception as e:
                print(a_text + ' has no core data', e)
                continue
            # 遍历所被选择的td，获取core data
            for td in td_tags:
                # print('-----' + td.get_text())
                td_text = td.get_text().split('：')
                stock_data[td_text[0]] = td_text[1]
                temp = td_text[0]

            # 去除退市股票以及不能采集到完整数据的股票
            if (len(self.Stocks) == 0) or (stock_data.keys() == self.Stocks[0].keys() and stock_data[temp] != '-'):
                self.Stocks.append(stock_data)

            counter = counter + 1
        print(self.Stocks[0])
        # 将数据写入excel
        self.write_excel()

    def write_excel(self):
        data = self.Stocks
        # 获取所有keys，也就是excel的第一行title的字段
        keys = data[0].keys()
        # 创建一个workbook
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建一个worksheet
        worksheet = workbook.add_sheet('sheet 1')

        # 写入sheet的第一行数据，也就是title
        col = 0
        for key in list(keys):
            # 参数对应 行, 列, 值
            worksheet.write(0, col, key)
            col = col + 1

        # 从sheet第二行开始，一行一行写数据
        row = 1
        for one_dict in data:
            col = 0
            for key in list(keys):
                worksheet.write(row, col, one_dict[key])
                col = col + 1
            row = row + 1
        # 保存文件
        workbook.save(self.fileName)
        print('spider ending ' + time.strftime('%Y/%m/%d  %I:%M:%S'))


if __name__ == '__main__':
    East()
    # print('R007(201001)'.split('(')[1].strip(')'))

    # 获取dict字段数据类型对象的第0个元素中所有的key
    # list_dict = [{'a': '_a', 'b': '_b'}, {'c': '_c', 'd': '_d'}]
    # list_keys = list_dict[0].keys()
    # print(list_keys)

    # 测试random
    # print(random.randint(1, 5) + random.random())
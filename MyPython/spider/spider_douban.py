"""
爬虫豆瓣电影top250
"""
import requests
import csv
from bs4 import BeautifulSoup


class Douban(object):
    def __init__(self, flag=False):
        self.douban_movie_top250 = 'https://movie.douban.com/top250'
        self.douban_book_top250 = 'https://book.douban.com/top250'
        # 一共有10页
        self.depth = 10
        # 存储定向信息
        self.books = []
        self.book_title = ['书名', '作者', '价格', '豆瓣评分', '一句话书评', '信息查阅']
        if flag:
            self.get_movie_top250()
        else:
            self.get_book_top250()

    def get_book_top250(self):
        for i in range(self.depth):
            s = i*25
            src = requests.get(self.douban_book_top250 + '?start=' + str(s) + '&filter=')
            src.encoding = 'utf-8'
            soup = BeautifulSoup(src.text, 'html.parser')
            td_tag = soup.find_all('td')
            for td in td_tag:
                if td.a.get('title') == None:
                    pass
                else:
                    name = td.a.get('title')
                    desc = td.p.text.split('/')
                    author = desc[0]
                    price = desc[-1]
                    rate = td.find('span', {'class': 'rating_nums'}).text
                    try:
                        inawords = td.find('span', {'class': 'inq'}).text
                    except:
                        inawords = ''
                    addr = td.a.get('href')
                    self.books.append([name, author, price, rate, inawords, addr])
        # 保存CSV
        with open('books_top250.csv', 'w', encoding='utf-8', newline='') as file:
            # 如果保存csv中文乱码(CSV是用UTF-8编码的，而EXCEL是ANSI编码，由于编码方式不一致导致出现乱码)，那就将csv以文本的形式打开，再将其编码修改为ANSI，保存即可
            writer = csv.writer(file)
            # 插入title
            writer.writerow(self.book_title)
            # 插入多行图书记录
            writer.writerows(self.books)
        # 保存TXT
        with open("books_top250.txt", "w", encoding="utf-8") as f:
            count = 0
            for each in self.books:
                count += 1
                booklist = "Top:{}\n书名:{}\n作者:{}\n定价:{}\n评分:{}\n一句话书评:{}\n信息查阅:{}\n"
                # print(booklist.format(count, each[0], each[1], each[2], each[3], each[4], each[5]))
                booklist = booklist.format(count, each[0], each[1], each[2], each[3], each[4], each[5])
                # f.write(str(count)+each+'\n')
                f.write(booklist)

    def get_movie_top250(self):
        # Demo: 获取豆瓣电影top250
        for i in range(self.depth):
            s = i*25
            src = requests.get(self.douban_movie_top250 + '?start=' + str(s) + "&filter=")
            # print(src.text)
            soup = BeautifulSoup(src.text, 'html.parser') # 指定解析器
            hd_div = soup.find_all('div', class_='hd')

            for item in hd_div:
                print(item.a.span.text)


if __name__ == '__main__':
    Douban(False)
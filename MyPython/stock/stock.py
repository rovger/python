import pandas as pd
import csv
import numpy as np
from sklearn2pmml import PMMLPipeline,sklearn2pmml
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.utils import shuffle

feature_sample = ['代码', '名称', '最新', '市盈率', '市净率', '所属行业', '流通市值', '营业总收入同比', '营业总收入', '营业利润', '每股收益', '流动资产', '利润总额', '每股净资产', '归属净利润同比', '未分配利润', '每股未分配利润', '销售毛利率', '总资产', '流动资产1', '固定资产', '无形资产', '资产负债比率', '公积金', '总负债', '流动负债', '长期负债', '每股公积金', '归属净利润']
data_sample = ['600000', '浦发银行', '11.05', '5.63', '0.76', '1', '3105亿', '1.31', '1271亿', '509亿', '1.47', '0', '509亿', '14.49', '3.14', '1285亿', '4.38', '0', '6.09万亿', '0', '254亿', '98.1亿', '92.41', '818亿', '5.63万亿', '0', '0', '2.79', '432亿']
result_titles = ['代码', '名称', '原总市值(亿)', '预测总市值(亿)', '市值差比率']
need_convert_index = [6, 8, 9, 11, 12, 15, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29]


def xlsx_to_csv(source_name):
    """
    format xlsx to csv
    :param source_name:
    :return:
    """
    print('converting xlsx to csv...')
    data_xlsx = pd.read_excel(source_name)
    data_xlsx.to_csv('data/stock.csv', encoding='utf-8')
    print('convert csv success.')


def convertor(temp):
    converted_arr = []
    for i in range(len(temp)):
        if '----' in temp:
            return None
        ele = temp[i]
        if i == 0 or i == 1:
            converted_arr.append(ele)
            continue
        if i in need_convert_index:
            if ele.find('万亿') != -1:
                ele = ele.replace('万亿', '')
                ele = float(ele)*10000
            elif ele.find('亿') != -1:
                ele = ele.replace('亿', '')
                ele = float(ele)
            elif ele.find('万') != -1:
                ele = ele.replace('万', '')
                ele = float(ele)/10000
            else:
                return None
        converted_arr.append(float(ele))
    return converted_arr


def load_csv_data(file_path):
    """
    analisys stock csv data
    :param file_path:
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        title = next(reader)
        # Title
        title_names = title[1:]
        print('sample:')
        print(title_names)

        # Content
        data, counter = [], 0
        for i, d in enumerate(reader):
            value = convertor(d[1:])
            if i == 0:
                print(value)
            if value is None:
                continue
            else:
                data.append(value)
                counter = counter+1
        print('valid data count: %d' % counter)
        return data, title_names, counter


def export_csv(file_name, feature_names, data):
    with open(file_name, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)

        # Column name
        writer.writerow(feature_names)
        # Rows values
        writer.writerows(data)


def core_process():
    """
    core process
    :return:
    """
    # xlsx_to_csv(source_name='data/20190111.xlsx')
    print('data cleaning & loading...')
    csv_data, title_names, n_samples = load_csv_data(file_path='data/stock.csv')
    # X_ not contain stock_code

    X_, y_ = [], []
    for i in range(n_samples):
        X_.append(csv_data[i][2:-1])
        y_.append(csv_data[i][-1])
        if i == 0:
            print(csv_data[i][2:-1])
            print(csv_data[i][-1])
    export_csv(file_name='stock_X_parse.csv', feature_names=title_names[2:-1], data=X_)
    # 乱序
    # X_ = shuffle(X_)
    x_train, x_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2)
    # n_train = int(n_samples * 0.8)
    # x_train = X_[0:n_train]
    # y_train = y_[0:n_train]
    # x_test = X_[n_train:]
    # y_test = y_[n_train:]

    # sklearn data standard
    print('x_train after standard...')
    # scaler = MinMaxScaler()
    # train_minmax = scaler.fit_transform(x_train)
    # test_minmax = scaler.fit_transform(x_test)
    # X_minmax = scaler.fit_transform(X_)
    # export_csv(file_name='stock_standard.csv', feature_names=title_names[2:-1], data=X_minmax)

    model = linear_model.LinearRegression()
    print('model train starting...')
    model.fit(x_train, y_train)

    # score
    print('score:', model.score(x_test, y_test))

    print('predict & export starting...')
    # predict
    result_data = []
    for i in range(n_samples):
        line_data, need_pred_data = [], []
        stock_no = csv_data[i][0]
        stock_name = csv_data[i][1]
        orig_value = float('%.2f' % y_[i])
        need_pred_data.append(X_[i])
        pred_values = model.predict(need_pred_data)
        pred_value = float('%.2f' % pred_values[0])
        value_gap = (pred_value-orig_value)/orig_value
        scope_rate = '%.2f%%' % (value_gap * 100)

        line_data.append(stock_no)
        line_data.append(stock_name)
        line_data.append(orig_value)
        line_data.append(pred_value)
        line_data.append(scope_rate)

        if i == 0:
            print(orig_value, need_pred_data, pred_value)

        result_data.append(line_data)

    # sort
    # result_data = sorted(result_data, key=lambda x: x[4], reverse=1)

    export_csv('predict_result.csv', feature_names=result_titles, data=result_data)
    print('process ending.')


if __name__ == '__main__':
    # xlsx_to_csv(source_name='data/20190111.xlsx')
    core_process()
    # print(data_sample)
    # print(convertor(data_sample))
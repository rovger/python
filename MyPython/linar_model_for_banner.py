#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Raymond Zhang <yuehzhang@ebay.com>
#
# Distributed under terms of the MIT license.

import sys
import pandas

"""
main function
"""
def main(input_path, output_path):
    data = pandas.read_csv(input_path)
    print(data['OFF_EBAY_VI'].apply(lambda x: 0.5 if x>= 100 else x/100.0 * 0.5))
    # print(data.dtypes)
    data['impression'] = data['SRP_IMPRSN'].apply( lambda x: 1.0 if x >=15000 else x/15000.0)
    data['click'] = data['SRP_VI'].apply(lambda x: 1.0 if x >= 1000 else x/1000.0)
    data['off_site_view'] = data['OFF_EBAY_VI'].apply(lambda x: 0.5 if x>= 100 else x/100.0 * 0.5)
    data['watch_score'] = data['WATCH'].apply(lambda x: 1.0 if x>= 100 else x/100.0)
    data['seller_lvl_score'] = data['slr_std'].apply(lambda x: 1.5 if x== 1  else 0.7 if x==2 else 0.1 if x== 3 else 0.0)
    data['deal_score'] = data['DD_VI'].apply(lambda x: 0.8 if x >= 50 or x == 0 else x/50.0 * 0.8 )
    data['shipping_score'] = data['free_int_shipping'].apply(lambda x: 0.2 if x==1 else 0.0)
    data['ctr'] = data['SRP_VI'] * 1.0 / data['SRP_IMPRSN']
    data['GMV'] = data['price'] * data['SI']
    data['transaction'] = data['GMV'].apply(lambda x: 3.0 if x >= 300.00 else 3.0 * x/300 )

    data['final_score'] = data['impression'] + data['click'] + data['off_site_view'] + data['watch_score'] + data['seller_lvl_score'] + data['deal_score'] + data['shipping_score'] + data['transaction'] + data['ctr']

    data = data.sort_values(['final_score'], ascending=False)
    # print(data)

    data.to_csv(output_path, sep='\t', encoding='utf-8')




def print_usage():
    print("usage: need to pass 3 args, first is input file path, second is index column name, third is output file path ")

"""
app entrance
"""
if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print("args number is %d" % len(sys.argv))
    #     print_usage()

    # input_path = sys.argv[1]
    # col_name = sys.argv[2]
    # output_path = sys.argv[3]
    sys.exit(main("Denmark_Finland_Norway_Sweden_12_a.csv", "Denmark.csv"))



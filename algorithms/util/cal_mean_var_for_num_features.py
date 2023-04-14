#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-18 11:07
# @Author  : shaoguang.csg
# @File    : cal_mean_var_for_numerical_features.py

# coding: utf-8
import sys

import os
import time
from pytalos.client import AsyncTalosClient


def output_file(file_name, data):
    print('result: ', data)
    with open(file_name, "w") as wf:
        for line in data:
            wf.write(line)


def get_hive_data(sql):
    username = "it_talos.waimai.cptming"
    password = "Waimai_ad"

    try:
        client = AsyncTalosClient(username=username, password=password)
        client.open_session()
        dsn = "hmart_waimaiad"
        engine = "hive"  # hive
        qid = client.submit(dsn=dsn, statement=sql, engine=engine, use_cache=False)

        while True:
            query_info = client.get_query_info(qid)
            if query_info['status'] == "FINISHED":
                finish = True
                break
            elif query_info["status"] in ["QUERY_TIMEOUT", "FAILED", "KILLED"] or query_info["status"].startswith("ERROR_"):
                finish = False
                break

            time.sleep(30)

        if finish:
            return client.fetch_result(qid, 0, 1)
        else:
            print(client.engine_log(qid))
            return None
    except Exception as e:
        print(e.message)


def update_mean_var(old_mean, old_var, old_num, mean, var, num, feature_cnt):
    # decay_ratio = 0.6
    new_mean, new_var, new_num = [], [], int(old_num) + int(num)
    for idx in range(feature_cnt):
        current_mean = (float(old_mean[idx]) * int(old_num) + float(mean[idx]) * int(num))/new_num
        # current_mean = float(old_mean[idx]) * decay_ratio + float(mean[idx]) * (1 - decay_ratio)
        new_mean.append(current_mean)

        current_var = ((int(num) - 1) * float(var[idx]) + (int(old_num) - 1) * float(old_var[idx])) / (new_num - 2)
        # current_var = float(old_var[idx]) * decay_ratio + float(var[idx]) * (1 - decay_ratio)
        new_var.append(current_var)
    return new_mean, new_var, new_num


def main():
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    output_prefix = sys.argv[3]
    table_name = sys.argv[4]
    task_type = sys.argv[5]
    high_features = sys.argv[6]
    low_features = sys.argv[7]

    if task_type in 'low':
        features = low_features
    
    else:
        features = high_features

    output_filename = "data/{task_type}_{prefix}.{current_date}".format(task_type=task_type,prefix=output_prefix, current_date=end_date)
    print('output_filename', output_filename)

    if os.path.exists(output_filename):
        #os.remove(output_filename)
        print("{filename} already exists!".format(filename=output_filename))
        return -1

    features = features.split(',')
    fea_avg = ['avg(cast({}[0] as double)+cast({}[1] as double)+cast({}[2] as double)+cast({}[3] as double))'.format(i,i,i,i) for i in features]
    fea_var = ['variance(cast({}[0] as double)+cast({}[1] as double)+cast({}[2] as double)+cast({}[3] as double))'.format(i,i,i,i) for i in features]
    sql = "select {0},{1},count(1) from {2} where dt between {3} and {4}".format(
        ','.join(fea_avg),
        ','.join(fea_var),
        table_name,
        start_date,
        end_date
    )
    print(sql)

    try_num = 1
    res = None
    while try_num <= 3:
        res = get_hive_data(sql)
        if res is None:
            try_num += 1
            print("try_num {try_num}".format(try_num=try_num))
            time.sleep(10)
        else:
            print("execute sql success")
            break

    if res is not None:
        print("start update mean and var {date} ... ".format(date=end_date))
        res = res['data'][0]
#        mean, var, num = res[0][:feature_cnt], res[0][feature_cnt:feature_cnt * 2], res[0][feature_cnt * 2]

        if os.path.exists(output_filename):
            print("{filename} already exists".format(filename=output_filename))
            return -1
        else:
            output_file(output_filename, '\t'.join([str(c) for c in res]))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-18 11:07
# @Author  : shaoguang.csg
# @File    : cal_mean_var_for_numerical_features.py

# coding: utf-8
import sys

import os
import time



def main():
    print("len: "+ str(len(sys.argv))+","+"\t".join(sys.argv))
    if len(sys.argv)>4:
        cur_state_cate_fea =("\""+"\",\"".join(sys.argv[1].split(","))+"\"").lower()
        next_state_cate_fea=("\""+"\",\"".join(sys.argv[3].split(","))+"\"").lower()
        cur_state_dynamic_fea = ("\""+"\",\"".join(sys.argv[2].split(","))+"\"").lower()
        next_state_dynamic_fea=("\""+"\",\"".join(sys.argv[4].split(","))+"\"").lower()

        file_name = sys.argv[5]

        json_content ="""
    {
        "features": [
            {
                "feature_name": "cur_state_cate_fea",
                "hive_feature_names": ["""+cur_state_cate_fea+"""],
                "concat_separator" : ",",
                "value_type": "string"
            },
            {
                "feature_name": "next_state_cate_fea",
                "hive_feature_names": ["""+next_state_cate_fea+"""],
                "concat_separator" : ",",
                "value_type": "string"
            },
            {
                "feature_name": "cur_state_dynamic_fea",
                "hive_feature_names": ["""+cur_state_dynamic_fea+"""],
                "concat_separator" : ",",
                "value_type": "float32"
            },
            {
                "feature_name": "next_state_dynamic_fea",
                "hive_feature_names": ["""+next_state_dynamic_fea+"""],
                "concat_separator" : ",",
                "value_type": "float32"
            }
        ]
    }
        """
    else:
        cur_state_dynamic_fea = ("\""+"\",\"".join(sys.argv[1].split(","))+"\"").lower()
        next_state_dynamic_fea=("\""+"\",\"".join(sys.argv[2].split(","))+"\"").lower()
        file_name = sys.argv[3]

        json_content ="""
    {
        "features": [
            {
                "feature_name": "cur_state_dynamic_fea",
                "hive_feature_names": ["""+cur_state_dynamic_fea+"""],
                "concat_separator" : ",",
                "value_type": "float32"
            },
            {
                "feature_name": "next_state_dynamic_fea",
                "hive_feature_names": ["""+next_state_dynamic_fea+"""],
                "concat_separator" : ",",
                "value_type": "float32"
            }
        ]
    }
        """
    print(json_content)
    f = open(file_name, 'w')
    f.write(json_content)
    f.close

if __name__ == '__main__':
    main()

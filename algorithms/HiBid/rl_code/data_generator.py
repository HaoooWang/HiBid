# coding=utf-8
import os, sys
import hashlib
import random
import tensorflow as tf
import functools
import math


class DataGenerator:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        self.file_names = []

        self.train_spec = None
        self.eval_spec = None

    def get_input_filenames(self):
        file_names = []
        data_dir = 'inputs'
        file_list = tf.gfile.ListDirectory(data_dir)
        for current_file_name in file_list:
            file_path = os.path.join(data_dir, current_file_name)
            file_names.append(file_path)
        self.logger.info('all files: %s' % file_names)
        random.shuffle(file_names)
        return file_names

    def build_data_spec(self):


        trainset_file_names = None
        validset_file_names = None

        train_config = self.config.copy()
        #        train_config['predict_trick_cate_fea'] = "" 
        self.logger.info("train_config %s", train_config)

        hooks = []
        
        self.train_input_fn = functools.partial(self.input_fn, trainset_file_names, train_config)
        self.train_spec = tf.estimator.TrainSpec(self.train_input_fn, hooks=hooks)

        self.logger.info('validset_file_names %s', validset_file_names)

        self.eval_input_fn = functools.partial(self.input_fn, validset_file_names, self.config)
        self.eval_spec = tf.estimator.EvalSpec(self.eval_input_fn,
                                               steps=100,
                                               start_delay_secs=6,
                                               throttle_secs=200)


    def input_fn(self, file_names, config):

        self.logger.info('file_names: %s', file_names)
        self.logger.info('batch_size: %s', config['batch_size'])

        def parse_fn(serialized_example):
            if config['task_type'] in "low":
                dynamic_list = config['low_state_dynamic_fea']
                cate_list = config['low_state_cate_fea']
            else:
                dynamic_list = config['high_state_dynamic_fea']
                cate_list = config['high_state_cate_fea']

            def input_process_feature_transfer(_parsed_features):
                
                cur_state_dynamic_feature = []
                for state in dynamic_list.split(','):
                    if len(state)<=1: continue
                    feature = tf.cast(_parsed_features[state],tf.float32) 
                    cur_state_dynamic_feature.append(tf.expand_dims(feature,axis = 1))
                cur_state_dynamic_feature = tf.concat(cur_state_dynamic_feature,axis = 1)

                cur_state_cate_feature = []
                for state in cate_list.split(','):
                    if len(state)<=1: continue
                    if state in ['poi_id']:
                        feature = tf.tile(_parsed_features[state],[1,4])
                    else:
                        feature = tf.as_string(_parsed_features[state])
                    cur_state_cate_feature.append(tf.expand_dims(feature,axis = 1))
                cur_state_cate_feature = tf.concat(cur_state_cate_feature,axis = 1)

                next_state_dynamic_feature = []
                for state in dynamic_list.split(','):
                    if len(state)<=1: continue
                    prefix='' if config['task_type'] in 'high' else 'next_'
                    feature = tf.cast(_parsed_features[prefix+state],tf.float32)
                    next_state_dynamic_feature.append(tf.expand_dims(feature,axis= 1))
                next_state_dynamic_feature = tf.concat(next_state_dynamic_feature,axis = 1)

                next_state_cate_feature = []
                for state in cate_list.split(','):
                    if len(state)<=1: continue
                    prefix='' if config['task_type'] in 'high' or state in 'poi_id' else 'next_'

                    if state in ['poi_id']:
                        feature = tf.tile(_parsed_features[prefix+state],[1,4])
                    else:
                        feature = tf.as_string(_parsed_features[prefix+state])
                    next_state_cate_feature.append(tf.expand_dims(feature,axis= 1))
                next_state_cate_feature = tf.concat(next_state_cate_feature,axis = 1)
                
                final_parsed_features = {
                    "poi_id": _parsed_features["poi_id"],
                    "real_budget":_parsed_features["real_budget"] if config['task_type'] in 'low' else _parsed_features["budget"],

                    "action": _parsed_features["action"],
                    "reward": tf.cast(_parsed_features['reward'], tf.float32),

                    "cur_state_cate_fea":cur_state_cate_feature,
                    "cur_state_dynamic_fea": cur_state_dynamic_feature,

                    "next_state_cate_fea": next_state_cate_feature,
                    "next_state_dynamic_fea": next_state_dynamic_feature
                }

                if config['task_type'] in "low":
                    final_parsed_features.update({
                        "prod": _parsed_features["prod"],
                        "pvid":_parsed_features["pvid"],
                        "cost": tf.cast(_parsed_features['final_charge'], tf.float32),
                        "origin_reward":_parsed_features["reward"],
                        "real_click":tf.cast(_parsed_features['real_click'], tf.float32), 
                        "real_cash":tf.cast(_parsed_features['real_csm'], tf.float32), 
                        "aimcpc":_parsed_features['aimcpc'],
                     
                        # high level 每一个mdp就是一条记录，不需要终止flag
                        "is_terminal": _parsed_features["is_terminal"]
                    })


                return final_parsed_features

            def bcorle_generate_lambda(_parsed_features, _config):
                """
                refer to: Zhang, Y., Tang, B., Yang, Q., An, D., Tang, H., Xi, C., ... & Xiong, F. (2021). BCORLE ($\lambda $): An Offline Reinforcement Learning and Evaluation Framework for Coupons Allocation in E-commerce Market. Advances in Neural Information Processing Systems, 34, 20410-20422.
                :param _parsed_features:
                :param _config:
                :return:
                """
                self.logger.info("use bcorle!!!!")
                bcorle_lambda_num = int(_config["bcorle_lambda_num"]) # 100
                bcorle_lambda_range = [float(s) for s in _config["bcorle_lambda_range"].split("_")]

                for key in _parsed_features:
                    col = _parsed_features[key]
                    shape = col.get_shape()
                    rank = len(shape)
                    if rank == 1:
                        col = tf.tile(col, [bcorle_lambda_num])
                    elif rank == 2:
                        col = tf.tile(col, [bcorle_lambda_num, 1])
                    elif rank == 3:
                        col = tf.tile(col, [bcorle_lambda_num, 1, 1])
                    _parsed_features[key] = col

                reward = _parsed_features["reward"] # is_click [B,4]
                
                bcorle_lambdas = tf.random.uniform(tf.shape(reward), bcorle_lambda_range[0], bcorle_lambda_range[1],dtype=tf.float32)

                _parsed_features["reward"] = reward - bcorle_lambdas* _parsed_features["cost"]
                _parsed_features["bcorle_lambdas"] = tf.expand_dims(bcorle_lambdas,axis=1) 
                return _parsed_features

            parsed_features = input_process_feature_transfer(serialized_example)
            if "use_bcorle" in config and config["use_bcorle"] == True and config["task_type"]=='low' and not config['ext_is_predict_serving'] and not config['task'] in 'predict':
                parsed_features = bcorle_generate_lambda(parsed_features, config)

            return parsed_features

        if file_names is None:
            self.logger.info(int(config['batch_size']))
            dataset = tf.data.AfoShmDataset(int(config['batch_size']),False)
            #dataset = dataset.batch(int(config['batch_size']), drop_remainder=False)
            dataset = dataset.map(parse_fn)
            dataset = dataset.prefetch(10)
        else:
            self.logger.info("error!!!!!!!!!!")
           

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


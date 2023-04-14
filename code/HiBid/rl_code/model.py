#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-12-17 16:06
# @Author  : shaoguang.csg
# @File    : model_dataset

import imp
from re import S
import re
import tensorflow as tf
import numpy as np
import json
from util.util import *
from tensorflow.contrib.lookup.lookup_ops import  get_mutable_dense_hashtable
from nets import BCQ_net, Mix_net
# from tensorflow.contrib.opt import HashAdamOptimizer
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
import os
import sys
from higher import MAModel
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 2)[0]  # 上2级目录
sys.path.append(config_path)
# from rl_easy_go_high.rl_code.main import Trainer

logger = set_logger()

def load_normalization_parameter(mean_var_filename, fea_num, prod_num=4,use_bcorle=False,save_prod=None):
    fea_mean = []
    fea_var = []
    logger.info("mean_var_filename%s",mean_var_filename)
    input_file = tf.gfile.Open(mean_var_filename)
    for line in input_file:
        splitted = line.strip().split("\t")
        for i in range (len(splitted)):
            if splitted[i] == "NULL":
                splitted[i] = 1.0

        for i in range(fea_num):
            fea_mean.append(float(splitted[i]))
        for i in range(fea_num):
            fea_var.append(float(splitted[i + fea_num]))
        break
    logger.info('num_fea_mean %s', fea_mean)
    logger.info('num_fea_var %s', fea_var)
  
    if save_prod is not None:
        return fea_mean,fea_var

    fea_mean = [[i for _ in range(prod_num)] for i in fea_mean]
    fea_var = [[i for _ in range(prod_num)] for i in fea_var]
    return fea_mean, fea_var




def make_coeff(num_heads):
    arr = np.random.uniform(low=0.0, high=1.0, size=num_heads)
    arr /= np.sum(arr)
    return arr
 


def Smooth_L1_Loss(labels, predictions, name, is_weights):
    with tf.variable_scope(name):
        diff = tf.abs(labels - predictions)
        less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)  # Bool to float32
        smooth_l1_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)
        return tf.reduce_mean(is_weights * smooth_l1_loss)  # get the average


def loss_function(y_logits, y_true):
    with tf.name_scope('loss'):
        cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true,
            logits=y_logits,
            name='xentropy'
        )
        loss = tf.reduce_mean(cross_entropy_loss, name='xentropy_mean')
    return loss


optimizer_mapping = {
    "adam": tf.train.AdamOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "rmsprop": lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.95)
}


def get_optimizer_by_name(name):
    if name not in optimizer_mapping.keys():
        logger.error("Unsupported {} currently, using sgd as default".format(name))
        return optimizer_mapping["sgd"]
    return optimizer_mapping[name]

@tf.function
def select_action(q, imt, threshold, use_bcq,lambda_vector=None,batch=False):
    return_shape = [-1,4] if batch else [-1,1]
    argmax_axis = 2 if batch else 1
    if lambda_vector is not None:
        new_q  = q - lambda_vector
    else:
        new_q = q 
    if use_bcq:
        imt = tf.exp(imt)
        imt = (imt / tf.reduce_max(imt, axis=1, keep_dims=True) >= threshold)
        imt = tf.cast(imt, dtype=tf.float32)
        return tf.reshape(tf.argmax(imt * new_q + (1. - imt) * -1e8, axis=argmax_axis), return_shape), tf.reshape(
            tf.reduce_max(imt * new_q + (1. - imt) * -1e8, axis=argmax_axis), return_shape)
    else:
        return tf.reshape(tf.argmax(new_q, axis=argmax_axis), return_shape), tf.reshape(
            tf.reduce_max(new_q, axis=argmax_axis), return_shape)


def model_fn(features, labels, mode, params):
    vocab_size = params['vocab_size']
    deep_layers = params['deep_layers'].split(',')
    learning_rate = params['learning_rate']
    update_interval = params['update_interval']
    num_action = params['num_action']
    embed_dim = params['embed_dim']
    ext_is_predict_serving = params['ext_is_predict_serving']
    threshold = params['threshold']
    i_loss_weight = params['i_loss_weight']
    i_regularization_weight = params['i_regularization_weight']
    q_loss_weight = params['q_loss_weight']
    gamma = params['gamma']
    num_heads = params['num_heads']
    use_rem = params['use_rem']
    use_bcq = params['use_bcq']
    use_bn = params['use_bn']

    high_cate_num = len(params['high_state_cate_fea'].split(","))
    high_dynamic_num = len(params['high_state_dynamic_fea'].split(","))
    low_cate_num = len(params['low_state_cate_fea'].split(","))
    low_dynamic_num = len(params['low_state_dynamic_fea'].split(","))

    prod_num = params['prod_num']
    use_adaptive = params['use_adaptive']
    use_batch_loss = params['use_batch_loss']
    use_bcorle = params['use_bcorle']
    use_mask = params['use_mask']

    save_prod = params.get("save_prod",None)

    task_type = params["task_type"]

    if use_adaptive:
        lambda_update_interval = params['lambda_update_interval']
        lambda_budgets_target =  [float(s) for s in params["lambda_budgets_target"].split("_")]
        auto_lambda_vector = tf.get_variable('auto_lambda_vector', shape=[4], dtype=tf.float32,
                                                initializer=tf.zeros_initializer(), trainable=False)
        target_auto_lambda_vector = tf.get_variable('target_auto_lambda_vector', shape=[4], dtype=tf.float32,
                                                        initializer=tf.zeros_initializer(), trainable=False)


    high_state_dynamic_fea_mean_var_filename = params['high_state_dynamic_fea_mean_var_filename']
    low_state_dynamic_fea_mean_var_filename = params['low_state_dynamic_fea_mean_var_filename']

    logger.info('params {}'.format(params))

    with tf.name_scope('model'):

        cur_state_cate_fea_col = features['cur_state_cate_fea']
        next_state_cate_fea_col = features['next_state_cate_fea']
        cur_state_dynamic_fea_col = features['cur_state_dynamic_fea']
        next_state_dynamic_fea_col = features['next_state_dynamic_fea']

        budgets = tf.cast(features['real_budget'],tf.float32)/100 # 分-》元 # B,4
        if task_type in 'low':
            real_click = tf.reduce_sum(tf.cast(features['real_click'],tf.float32),axis=1,keepdims=True )# B,1
            real_cash = tf.reduce_sum(tf.cast(features['real_cash'],tf.float32),axis=1,keepdims=True)/100  # B,1 分->元
            aimcpc = tf.reduce_sum(tf.cast(features['aimcpc'],tf.float32),axis=1,keepdims=True)/100  # B,1 分->元
        reward = tf.cast(features['reward'], tf.float32)
        action_col = tf.cast(features['action'],tf.int64)

        #reward = tf.Print(reward, [reward], "#reward", summarize=1000)


        with tf.name_scope('variable'):
            embedding_matrix = get_mutable_dense_hashtable(key_dtype=tf.int64,
                                            value_dtype=tf.float32,
                                            shape=tf.TensorShape([embed_dim]),
                                            name="embed_table",
                                            initializer=tf.truncated_normal_initializer(0.0, 1e-2),
                                            shard_num=2)
     

        with tf.name_scope('input'):
            if ext_is_predict_serving and mode == tf.estimator.ModeKeys.PREDICT:
                cur_state_cate_fea_col = tf.as_string(cur_state_cate_fea_col)
                next_state_cate_fea_col = tf.as_string(next_state_cate_fea_col)

                low_prefix = tf.constant([str(i)+"_low"  for i in range(low_cate_num)]) 
                high_prefix = tf.constant([str(i)+"_high" for i in range(high_cate_num)]) 
                high_cate_feature = get_hash_cate_feature(cur_state_cate_fea_col[:,:high_cate_num],high_prefix,embedding_matrix,high_cate_num,vocab_size,embed_dim,1)
                high_next_cate_feature = get_hash_cate_feature(next_state_cate_fea_col[:,:high_cate_num],high_prefix,embedding_matrix,high_cate_num,vocab_size,embed_dim,1)
                low_cate_feature = get_hash_cate_feature(cur_state_cate_fea_col,low_prefix,embedding_matrix,low_cate_num,vocab_size,embed_dim,1)
                low_next_cate_feature = get_hash_cate_feature(next_state_cate_fea_col,low_prefix,embedding_matrix,low_cate_num,vocab_size,embed_dim,1)

                high_cate_feature = tf.squeeze(high_cate_feature,axis=-1)
                high_next_cate_feature = tf.squeeze(high_next_cate_feature,axis=-1)
                low_cate_feature = tf.squeeze(low_cate_feature,axis=-1)
                low_next_cate_feature = tf.squeeze(low_next_cate_feature,axis=-1)

                # cur_state_dynamic_fea_col = tf.tile(tf.expand_dims(cur_state_dynamic_fea_col,axis=-1),[1,1,4])
                # next_state_dynamic_fea_col = tf.tile(tf.expand_dims(next_state_dynamic_fea_col,axis=-1),[1,1,4])
                # budgets = tf.tile(budgets,[1,4])

                #real_cur_state_dynamic_fea_col= tf.stack([cur_state_dynamic_fea_col[:,-28:-21,0],cur_state_dynamic_fea_col[:,-21:-14,1], cur_state_dynamic_fea_col[:,-14:-7,2],cur_state_dynamic_fea_col[:,-7:,3]],axis=-1)
                if save_prod==0:
                    cur_state_dynamic_fea_col = tf.concat([cur_state_dynamic_fea_col[:,0:-28],cur_state_dynamic_fea_col[:,-28:-21]],axis=1)
                    next_state_dynamic_fea_col = tf.concat([next_state_dynamic_fea_col[:,0:-28],next_state_dynamic_fea_col[:,-28:-21]],axis=1)
                elif save_prod==1:
                    cur_state_dynamic_fea_col = tf.concat([cur_state_dynamic_fea_col[:,0:-28],cur_state_dynamic_fea_col[:,-21:-14]],axis=1)
                    next_state_dynamic_fea_col = tf.concat([next_state_dynamic_fea_col[:,0:-28],next_state_dynamic_fea_col[:,-21:-14]],axis=1)
                elif save_prod==2:
                    cur_state_dynamic_fea_col = tf.concat([cur_state_dynamic_fea_col[:,0:-28],cur_state_dynamic_fea_col[:,-14:-7]],axis=1)
                    next_state_dynamic_fea_col = tf.concat([next_state_dynamic_fea_col[:,0:-28],next_state_dynamic_fea_col[:,-14:-7]],axis=1)
                elif save_prod==3:
                    cur_state_dynamic_fea_col = tf.concat([cur_state_dynamic_fea_col[:,0:-28],cur_state_dynamic_fea_col[:,-7:]],axis=1)
                    next_state_dynamic_fea_col = tf.concat([next_state_dynamic_fea_col[:,0:-28],next_state_dynamic_fea_col[:,-7:]],axis=1)

                # real_next_state_dynamic_fea_col= tf.stack([next_state_dynamic_fea_col[:,-28:-21,0],next_state_dynamic_fea_col[:,-21:-14,1], next_state_dynamic_fea_col[:,-14:-7,2],next_state_dynamic_fea_col[:,-7:,3]],axis=-1)
                # next_state_dynamic_fea_col = tf.concat([next_state_dynamic_fea_col[:,0:-28,:],real_next_state_dynamic_fea_col],axis=1)

            else:
                # poi static feature
                low_prefix = tf.constant([[str(i)+"_low" for _ in range(4)]  for i in range(low_cate_num)]) 
                high_prefix = tf.constant([[str(i)+"_high" for _ in range(4)]  for i in range(high_cate_num)]) 

                if task_type in 'high':
                    high_cate_feature = get_hash_cate_feature(cur_state_cate_fea_col,high_prefix,embedding_matrix,high_cate_num,vocab_size,embed_dim,prod_num)
                    high_next_cate_feature = get_hash_cate_feature(next_state_cate_fea_col,high_prefix,embedding_matrix,high_cate_num,vocab_size,embed_dim,prod_num)

                if task_type in 'low':
                    # 上层只有poi_id是离散特征
                    high_cate_feature = get_hash_cate_feature(cur_state_cate_fea_col[:,:high_cate_num],high_prefix,embedding_matrix,high_cate_num,vocab_size,embed_dim,prod_num)
                    high_next_cate_feature = get_hash_cate_feature(next_state_cate_fea_col[:,:high_cate_num],high_prefix,embedding_matrix,high_cate_num,vocab_size,embed_dim,prod_num)

                    low_cate_feature = get_hash_cate_feature(cur_state_cate_fea_col,low_prefix,embedding_matrix,low_cate_num,vocab_size,embed_dim,prod_num)
                    low_next_cate_feature = get_hash_cate_feature(next_state_cate_fea_col,low_prefix,embedding_matrix,low_cate_num,vocab_size,embed_dim,prod_num)


        ## TODO 不同渠道的特征划分，由于real_time特征每个渠道全喂了，每个渠道只需要一部分
        ##  shape [B,S,4] -> [B,S-12,4]

        high_model = MAModel(deep_layers=deep_layers,num_action=num_action,optimizer='adam',variable_scope='high_controller',learning_rate=learning_rate,
                                use_bn=use_bn,use_bcq=use_bcq,threshold=threshold,use_rem=use_rem,num_heads=num_heads,logger=logger)

        low_model = MAModel(deep_layers=deep_layers,num_action=num_action,optimizer='adam',variable_scope='low_controller',learning_rate=learning_rate,
                                use_bn=use_bn,use_bcq=use_bcq,threshold=threshold,use_rem=use_rem,num_heads=num_heads,logger=logger,use_mask=use_mask)

        low_model_cost = MAModel(deep_layers=deep_layers,num_action=num_action,optimizer='adam',variable_scope='low_controller_cost',learning_rate=learning_rate,
                        use_bn=use_bn,use_bcq=use_bcq,threshold=threshold,use_rem=use_rem,num_heads=num_heads,logger=logger)
        
        low_model_click = MAModel(deep_layers=deep_layers,num_action=num_action,optimizer='adam',variable_scope='low_controller_click',learning_rate=learning_rate,
                        use_bn=use_bn,use_bcq=use_bcq,threshold=threshold,use_rem=use_rem,num_heads=num_heads,logger=logger)
  
        lambda_helper = LambdaHelper(click_model=low_model_click,cost_model=low_model_cost,save_prod=save_prod)
        global_step = tf.train.get_or_create_global_step()
        
        with tf.name_scope('dict'):
            high_state_dynamic_mean_vec, high_state_dynamic_var_vec = load_normalization_parameter(
                high_state_dynamic_fea_mean_var_filename,
                high_dynamic_num,
                prod_num,
                use_bcorle=use_bcorle,
                save_prod=save_prod
            )
            if task_type in 'low':
                low_state_dynamic_mean_vec, low_state_dynamic_var_vec = load_normalization_parameter(
                    low_state_dynamic_fea_mean_var_filename,
                    low_dynamic_num,
                    prod_num,
                    use_bcorle=use_bcorle,
                    save_prod=save_prod
                )
        # 目前上层不需要lambda求解,lambda_vector_high已废用
        if use_adaptive and task_type in 'high':
            lambda_vector_high = get_calibration_vector(auto_lambda_vector) 
        else:
            lambda_vector_high = [None,None,None,None]
            lambda_vector_low = [None,None,None,None] # lambda泛化方案求解lambda
        
        # if (mode == tf.estimator.ModeKeys.PREDICT or ext_is_predict_serving or not use_bcorle) and task_type not in 'high':
        #     lambda_vector=tf.tile(tf.reshape(tf.ones_like(features['prod'],dtype=tf.float32),[-1,1,1]),[1,1,4])
        #     cur_state_dynamic_fea_col = tf.concat([cur_state_dynamic_fea_col,lambda_vector],axis=1)
        #     next_state_dynamic_fea_col = tf.concat([next_state_dynamic_fea_col,lambda_vector],axis=1)
        #     use_bcorle = True
    
        
        if task_type in 'high':
            cur_state_dynamic_fea_col = get_normalized_feature(cur_state_dynamic_fea_col,high_state_dynamic_mean_vec,high_state_dynamic_var_vec)
            next_state_dynamic_fea = get_normalized_feature(next_state_dynamic_fea_col,high_state_dynamic_mean_vec,high_state_dynamic_var_vec)

            current_state = tf.concat([high_cate_feature,cur_state_dynamic_fea_col],axis=1)
            next_state = tf.concat([high_next_cate_feature,next_state_dynamic_fea],axis=1)

            is_terminal = tf.cast(tf.concat([tf.zeros_like(action_col[:,:3]),tf.ones_like(tf.expand_dims(action_col[:,3],axis=1))],axis=1),tf.float32)

            best_actions,best_action_q,total_q_logits,total_q_imts,total_q_i = high_model.forward(current_state,lambda_vector_high,'main')
            # 上层的next
            temp_next = tf.concat([current_state[:,:,1:],tf.expand_dims(current_state[:,:,0],axis=-1)],axis=-1)
            _,next_best_action_q,_,_,_= high_model.forward(temp_next,lambda_vector_high,'target')

            target_q = reward + (1 - is_terminal) * gamma * next_best_action_q 
            selected_q = high_model.get_actions_qvalue(total_q_logits,action_col)

            if use_batch_loss:
                with tf.name_scope("constraint"):
                    constraint_weight = [float(s) for s in params["constraint_loss_weight"].split("_")]
                
                    dj_actions = get_approx_action(total_q_logits[:,0,:],total_q_imts[:,0,:],total_q_i[:,0,:],use_bcq,threshold)
                    bj_actions = get_approx_action(total_q_logits[:,1,:],total_q_imts[:,1,:],total_q_i[:,1,:],use_bcq,threshold)
                    ss_actions = get_approx_action(total_q_logits[:,2,:],total_q_imts[:,2,:],total_q_i[:,2,:],use_bcq,threshold)
                    push_actions =get_approx_action(total_q_logits[:,3,:],total_q_imts[:,3,:],total_q_i[:,3,:],use_bcq,threshold)
                    
                    actions_all = tf.concat([dj_actions,bj_actions,ss_actions,push_actions],axis=1)
                    
                    # 
                    prod_cost = tf.reduce_sum(action_to_budgets(actions_all,budgets),axis=0)*0.01 # [B,4] * B[B,4]
                    prod_target = tf.reduce_sum(action_to_budgets(action_col,budgets),axis=0)*0.01 # [4]
                    
                    dj_loss = get_constraint_loss(target=prod_target[0], pred=prod_cost[0])
                    bj_loss = get_constraint_loss(target=prod_target[1], pred=prod_cost[1])
                    ss_loss = get_constraint_loss(target=prod_target[2], pred=prod_cost[2])
                    push_loss = get_constraint_loss(target=prod_target[3], pred=prod_cost[3])

                    tf.Print(prod_cost, ["prod_cost:",prod_cost,"prod_target",prod_target], "#prod_cost", summarize=100)
                    print(prod_cost)
                    print(prod_target)
                    all_constraint_loss = dj_loss*constraint_weight[0]+bj_loss*constraint_weight[1]+ss_loss*constraint_weight[2]+push_loss*constraint_weight[3]
                    all_constraint_loss = tf.Print(all_constraint_loss, [all_constraint_loss], "#all_constraint_loss", summarize=100)

            if use_adaptive:
                update_lambda_op= update_lambda_vector_multi_step_no_dependency(params,auto_lambda_vector,[total_q_logits,total_q_imts,total_q_i],total_budgets,lambda_budgets_target,use_bcq)
             
                lambda_pos_counter = tf.get_variable(
                    'lambda_pos_counter',
                    shape=[],
                    dtype=tf.int64,
                    initializer=tf.zeros_initializer(),
                    trainable=False
                )
                lambda_neg_counter = tf.get_variable(
                    'lambda_neg_counter',
                    shape=[],
                    dtype=tf.int64,
                    initializer=tf.zeros_initializer(),
                    trainable=False
                )
                update_target_lambda_cond = is_update_target_qnet(global_step, lambda_update_interval)

                update_target_lambda_op = tf.cond(
                    update_target_lambda_cond,
                    true_fn=lambda: update_target_lambda_vector(auto_lambda_vector, target_auto_lambda_vector, lambda_pos_counter),
                    false_fn=lambda: do_nothing(lambda_neg_counter)
                )
           
        
        elif task_type in 'low':
            cur_state_dynamic_fea_col = get_normalized_feature(cur_state_dynamic_fea_col,low_state_dynamic_mean_vec,low_state_dynamic_var_vec)
            next_state_dynamic_fea = get_normalized_feature(next_state_dynamic_fea_col,low_state_dynamic_mean_vec,low_state_dynamic_var_vec)

            if save_prod is None:
                higher_actions,_,_,_,_ = high_model.forward(tf.concat([high_cate_feature,cur_state_dynamic_fea_col[:,:high_dynamic_num,:]],axis=1),lambda_vector_high,'main',save_prod=save_prod)
                higher_actions_emb = tf.reshape(tf.as_string(higher_actions),[-1,1,4])
                budget_prefix = tf.constant([[str(i)+"_budget" for _ in range(4)]]) 
                higher_actions_emb = get_hash_cate_feature(higher_actions_emb,budget_prefix,embedding_matrix,1,vocab_size,embed_dim,prod_num)
                higher_budgets = action_to_budgets(higher_actions,budgets) # [B,4]
            else:
                higher_actions,_,_,_,_ = high_model.forward(tf.concat([high_cate_feature,cur_state_dynamic_fea_col[:,:high_dynamic_num]],axis=1),lambda_vector_high,'main',save_prod=save_prod)
                higher_actions_emb = tf.reshape(tf.as_string(higher_actions),[-1,1])
                budget_prefix = tf.constant([str(save_prod)+"_budget"]) 
                higher_actions_emb = get_hash_cate_feature(higher_actions_emb,budget_prefix,embedding_matrix,1,vocab_size,embed_dim,1)
                higher_actions_emb = tf.reshape(higher_actions_emb,[-1,embed_dim])
                higher_budgets = action_to_budgets(higher_actions,budgets) # [B,4]
          
            if mode == tf.estimator.ModeKeys.PREDICT:
                lambda_state = tf.concat([low_cate_feature,cur_state_dynamic_fea_col,higher_actions_emb],axis=1)
                # lambda_s = tf.map_fn(generate_lambda, lambda_state, tf.float32, name="get_lambda")
                # new_state = tf.tile(tf.expand_dims(lambda_state,axis=1),[1,100,1,1])
                # logger.info("lambda_s:{},new_state:{}".format(tf.shape(lambda_s),tf.shape(new_state)))
                # lambda_state = tf.concat([new_state,lambda_s],axis=2) #[B,100,s,4]
                # [B,S,4] #[B,4]
                bcorle_lambdas = tf.map_fn(lambda_helper.get_optimal_lambda, (lambda_state,higher_budgets), tf.float32, name="lambda_search")
            else:
                if use_bcorle:
                    logger.info("use bcorle!")
                    bcorle_lambdas=features['bcorle_lambdas']
                    #bcorle_lambdas = tf.Print(bcorle_lambdas, [bcorle_lambdas], "#bcorle_lambdas", summarize=1000)
                else:
                    bcorle_lambdas = tf.zeros([tf.shape(cur_state_dynamic_fea_col)[0],1,4],dtype=tf.float32)

            current_state = tf.concat([low_cate_feature,cur_state_dynamic_fea_col,higher_actions_emb,bcorle_lambdas],axis=1)
            next_state = tf.concat([low_next_cate_feature,next_state_dynamic_fea,higher_actions_emb,bcorle_lambdas],axis=1)
            if save_prod is None:
                current_state = tf.reshape(current_state,[-1,104,4])
            else:
                current_state = tf.reshape(current_state,[-1,104])
            
            is_terminal = tf.tile(tf.cast(features['is_terminal'], tf.float32),[1,prod_num])

            logger.info("current_state{}".format(current_state.get_shape()))
    
            _,_,total_q_logits_cost,_,_ = low_model_cost.forward(current_state,None,'main',None,save_prod=save_prod)
            _,_,next_total_q_logits_cost,_,_= low_model_cost.forward(next_state,None,'target',None,save_prod=save_prod)

            _,_,total_q_logits_click,_,_ = low_model_click.forward(current_state,None,'main',None,save_prod=save_prod)
            _,_,next_total_q_logits_click,_,_= low_model_click.forward(next_state,None,'target',None,save_prod=save_prod)
            
            if use_mask:
                #real_click=None,real_cash=None,qv=None,qc=None,aim_cpc=None
                real_click = tf.tile(tf.reshape(real_click,[-1,1,1]),[1,4,20]) #[B,4,20]
                real_cash = tf.tile(tf.reshape(real_cash,[-1,1,1]),[1,4,20])
                aim_cpc = tf.tile(tf.reshape(aimcpc,[-1,1,1]),[1,4,20])
                total_q_logits_cost = tf.clip_by_value(total_q_logits_cost, clip_value_min=0, clip_value_max=1000)
                virtual_cpc = (real_cash+total_q_logits_click)/(real_click+total_q_logits_cost+1e-3)
                virtual_cpc = tf.clip_by_value(virtual_cpc, clip_value_min=0, clip_value_max=1000)
                valid_action = tf.less(virtual_cpc,aim_cpc)
                valid_action = tf.stop_gradient(valid_action)
            else:
                valid_action = None
          
            best_actions,best_action_q,total_q_logits,total_q_imts,total_q_i = low_model.forward(current_state,None,'main',valid_action,save_prod=save_prod)
            next_best_action,next_best_action_q,_,_,_= low_model.forward(next_state,None,'target',valid_action,save_prod=save_prod)
            next_best_action = tf.stop_gradient(next_best_action)

            prod = tf.cast(features['prod'] , tf.int64)
            prod = tf.subtract(prod, 1)
    
            if mode == tf.estimator.ModeKeys.PREDICT:
                if ext_is_predict_serving == 1:
                    tf.identity(tf.cast(best_actions, dtype=tf.float32))
                    result = tf.concat([tf.cast(best_actions,dtype=tf.float32),best_action_q], axis=1, name="output_action")
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=result)
                else:
                    prod = tf.cast(tf.reshape(prod,[-1]),tf.int64)
                    prod_one_hot = tf.one_hot(indices=prod, depth=4)

                    best_actions = tf.cast(tf.reshape(best_actions,[-1,4]), dtype=tf.float32)
                    best_action_q = tf.reshape(best_action_q,[-1,4])

                    best_actions = tf.reduce_sum(tf.multiply(prod_one_hot,best_actions),axis=1,keepdims=True)
                    best_action_q = tf.reduce_sum(tf.multiply(prod_one_hot,best_action_q),axis=1,keepdims=True)
                    logger.info("best_actions:{},best_action_q:{}".format(best_actions.get_shape(),best_action_q.get_shape()))

                    predictions = {
                        "action": best_actions,
                        "qvalue": best_action_q,
                        "poi_id": features['poi_id'],
                        "cur_action": action_col,
                        "pvid":features["pvid"],
                        "prod":features['prod']
                    }
                    logger.info("offline predict")
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

         
            h_index = tf.reshape(prod,[-1,1])
            line = tf.cast(tf.reshape(tf.range(tf.shape(prod)[0]),[-1,1]),tf.int64)
            index = tf.concat([line,h_index],axis = 1)

            # q_logits [B,4,20] actions [B,4]
            target_q = reward + (1 - is_terminal) * gamma * next_best_action_q 
            selected_q = low_model.get_actions_qvalue(total_q_logits,action_col)

            cost  = tf.cast(features['cost'],tf.float32)
            origin_reward =  tf.cast(features['origin_reward'],tf.float32)

        
            target_q_cost = cost + (1 - is_terminal) * gamma * low_model_cost.get_actions_qvalue(next_total_q_logits_cost,next_best_action) 
            selected_q_cost = low_model_cost.get_actions_qvalue(total_q_logits_cost,action_col)

            target_q_click = origin_reward + (1 - is_terminal) * gamma * low_model_click.get_actions_qvalue(next_total_q_logits_click,next_best_action) 
            selected_q_click = low_model_click.get_actions_qvalue(total_q_logits_click,action_col)

            with tf.name_scope('output'):
                logger.info("total_q_logits:{},total_q_imts:{},selected_q:{},target_q:{},best_actions:{},best_action_q:{}".format(
                    total_q_logits.get_shape(),total_q_imts.get_shape(),selected_q.get_shape(),target_q.get_shape(),best_actions.get_shape(),best_action_q.get_shape()
                ))
                
                action_col = tf.squeeze(tf.gather_nd(action_col,index))
                total_q_logits = tf.squeeze(tf.gather_nd(total_q_logits,index))
                total_q_imts = tf.squeeze(tf.gather_nd(total_q_imts,index))
                best_actions = tf.squeeze(tf.gather_nd(best_actions,index))
                best_action_q = tf.squeeze(tf.gather_nd(best_action_q,index))

                selected_q = tf.squeeze(tf.gather_nd(selected_q,index))
                target_q = tf.squeeze(tf.gather_nd(target_q,index))

                selected_q_cost = tf.squeeze(tf.gather_nd(selected_q_cost,index))
                target_q_cost = tf.squeeze(tf.gather_nd(target_q_cost,index))

                selected_q_click = tf.squeeze(tf.gather_nd(selected_q_click,index))
                target_q_click = tf.squeeze(tf.gather_nd(target_q_click,index))
                logger.info("total_q_logits:{},total_q_imts:{},selected_q:{},target_q:{},best_actions:{},best_action_q:{}".format(
                    total_q_logits.get_shape(),total_q_imts.get_shape(),selected_q.get_shape(),target_q.get_shape(),best_actions.get_shape(),best_action_q.get_shape()
                ))

        is_weights =tf.cast(tf.ones_like(action_col), tf.float32)  # 为后续增加权重做准备，当前是1

        error = tf.reduce_mean(tf.abs(target_q - selected_q)) #[256,4] vs. [256]
        q_loss = q_loss_weight * Smooth_L1_Loss(target_q, selected_q, "loss", is_weights)
        
        if task_type in 'low':
            error_cost = tf.reduce_mean(tf.abs(target_q_cost - selected_q_cost)) 
            q_loss_cost = q_loss_weight * Smooth_L1_Loss(target_q_cost, selected_q_cost, "cost_loss", is_weights)

            error_click = tf.reduce_mean(tf.abs(target_q_click - selected_q_click)) 
            q_loss_click = q_loss_weight * Smooth_L1_Loss(target_q_click, selected_q_click, "click_loss", is_weights)

            tf.Print(q_loss_cost, ["q_loss_cost:",q_loss_cost,"q_loss_click",q_loss_click], "#prod_cost", summarize=100)
            
        else:
            q_loss_cost = 0
            q_loss_click = 0

        all_loss = q_loss + q_loss_cost + q_loss_click
        
        if use_bcq:
            i_loss = i_loss_weight * tf.reduce_mean(
                tf.multiply(
                    is_weights,
                    tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits( # logits -> [batch_size, num_classes]，label -> [batch_size, 1]
                        labels=action_col, logits=total_q_logits), axis=1)
                )
            )
            i_reg_loss = i_regularization_weight * tf.reduce_mean(
                tf.multiply(
                    is_weights,
                    tf.reduce_mean(tf.pow(total_q_logits, 2), axis=1)
                )
            )
            all_loss +=  i_loss + i_reg_loss
            logger.info('i_loss {}'.format(i_loss))
            logger.info('i_reg_loss {}'.format(i_reg_loss))
            tf.summary.scalar('i_loss', i_loss)
            tf.summary.scalar('i_reg_loss', i_reg_loss)

        if use_batch_loss and task_type in 'high':
            all_loss += all_constraint_loss
            tf.summary.scalar('all_constraint_loss', all_constraint_loss)
            logger.info('all_constraint_loss {}'.format(all_constraint_loss))
        
        logger.info('q_loss {}'.format(q_loss))
        logger.info('all_loss {}'.format(all_loss))
        tf.summary.scalar('total_action_qvalue', tf.reduce_mean(total_q_logits))
        tf.summary.scalar('best_action_qvalue', tf.reduce_mean(best_action_q))
        tf.summary.scalar('reward', tf.reduce_mean(reward))
        tf.summary.scalar('loss', all_loss)
        tf.summary.scalar('q_loss', q_loss)
        tf.summary.scalar('abs_error', error)

        if task_type in 'low':
            logger.info('q_loss_cost {}'.format(q_loss_cost))
            tf.summary.scalar('q_loss_cost', q_loss_cost)
            tf.summary.scalar('abs_error_cost', error_cost)

            logger.info('q_loss_click {}'.format(q_loss_click))
            tf.summary.scalar('q_loss_click', q_loss_click)
            tf.summary.scalar('abs_error_click', error_click)

        main_qnet_var = []
        target_qnet_var = []
        for i in range(prod_num):
            main_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller/agent{}/main_{}_net'.format(task_type,i,i)))
            target_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller/agent{}/target_{}_net'.format(task_type,i,i)))

            if task_type in 'low':
                main_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller_cost/agent{}/main_{}_net'.format(task_type,i,i)))
                target_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller_cost/agent{}/target_{}_net'.format(task_type,i,i)))
                main_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller_click/agent{}/main_{}_net'.format(task_type,i,i)))
                target_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller_click/agent{}/target_{}_net'.format(task_type,i,i)))
        
        all_main_var = main_qnet_var[0]+main_qnet_var[1]+main_qnet_var[2]+main_qnet_var[3]
        all_target_var = target_qnet_var[0]+target_qnet_var[1]+target_qnet_var[2]+target_qnet_var[3]

      
        pos_counter = tf.get_variable(
            'pos_counter',
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        neg_counter = tf.get_variable(
            'neg_counter',
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False
        )


        update_target_qnet_cond = is_update_target_qnet(global_step, update_interval)

        update_target_qnet_op = tf.cond(
            update_target_qnet_cond,
            true_fn=lambda: update_target_qnet(all_main_var, all_target_var, pos_counter),
            false_fn=lambda: do_nothing(neg_counter)
        )
        tf.summary.scalar('pos_counter', pos_counter)
        tf.summary.scalar('neg_counter', neg_counter)
        tf.summary.scalar('counter_ratio', pos_counter / (neg_counter + 1))

        update_op = [update_target_qnet_op]
     
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.control_dependencies(update_op):
                var_diff = tf.add_n(
                    [tf.reduce_mean(tf.squared_difference(t, e)) for t, e in zip(all_main_var, all_target_var)])
                tf.summary.scalar('var_diff', tf.reduce_mean(var_diff))
               
                if task_type in 'high':
                    train_op = high_model.get_train_op(global_step,all_loss)
                elif task_type in 'low': 
                    train_op = low_model.get_train_op(global_step,all_loss)
             
                #train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(all_loss,global_step=global_step)
            return tf.estimator.EstimatorSpec(mode=mode, loss=all_loss, train_op=train_op)

        return tf.estimator.EstimatorSpec(mode=mode, loss=all_loss)


def is_update_target_qnet(global_step, update_interval):
    ret = tf.equal(tf.mod(global_step, tf.constant(update_interval, dtype=tf.int64)), tf.constant(0, dtype=tf.int64))
    tf.summary.scalar("is_update_target_qnet", tf.cast(ret, tf.int32))
    return ret


def update_target_qnet(main_qnet_var, target_qnet_var, pos_counter):
    logger.info("all trainable vars: {}".format(tf.trainable_variables()))
    logger.info("main qnet vars: {}".format(main_qnet_var))
    logger.info("target qnet vars: {}".format(target_qnet_var))

    ops = [tf.assign_add(pos_counter, 1)]
    ops.extend([tf.assign(t, e) for t, e in zip(target_qnet_var, main_qnet_var)])
    update_op = tf.group(ops)
    return update_op


def do_nothing(neg_counter):
    ops = [tf.assign_add(neg_counter, 1)]
    return tf.group(ops)


def train_function(loss, optimizer, global_step, learning_rate=0.001):
    with tf.name_scope('optimizer'):
        opt = get_optimizer_by_name(optimizer)(learning_rate)
    return opt.minimize(loss, global_step=global_step)



def update_target_lambda_vector(lambda_vector, target_lambda_vector, pos_counter):
    ops = [tf.assign_add(pos_counter, 1)]
    ops.extend([tf.assign(target_lambda_vector, lambda_vector)])
    update_op = tf.group(ops)
    return update_op


def is_update_auto_lambda(global_step, update_interval):
    ret = tf.equal(tf.mod(global_step, tf.constant(update_interval, dtype=tf.int64)), tf.constant(0, dtype=tf.int64))
    tf.summary.scalar("is_update_target_lambda", tf.cast(ret, tf.int32))
    return ret

# 生成线上tfserving的输入向量的格式
# 注意，placeholder的name需要和线上一致！！！！
def export_serving_model_input(params):
    low_cate_num = len(params['low_state_cate_fea'].split(","))
    low_dynamic_num = len(params['low_state_dynamic_fea'].split(","))

    feature_spec = {
        "pvid": tf.placeholder(dtype=tf.string, shape=[None, 1], name='pvid'),
        "prod": tf.placeholder(dtype=tf.int64, shape=[None, 1], name='prod'), #TODO
        "poi_id": tf.placeholder(dtype=tf.int64, shape=[None, 1], name='poi_id'),
        "real_budget": tf.placeholder(dtype=tf.float32, shape=[None, 1], name='real_budget'),

        "real_click": tf.placeholder(dtype=tf.float32, shape=[None, 4], name='real_click'),
        "real_cash": tf.placeholder(dtype=tf.float32, shape=[None, 4], name='real_cash'),
        "aimcpc": tf.placeholder(dtype=tf.float32, shape=[None, 1], name='aimcpc'),
      
        "is_terminal": tf.placeholder(dtype=tf.float32, shape=[None, 1], name='is_terminal'),
        "reward": tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward'),
        "action": tf.placeholder(dtype=tf.int64, shape=[None, 1], name='action'),

        "cur_state_cate_fea": tf.placeholder(dtype=tf.int64, shape=[None, low_cate_num], name='cur_state_cate_fea'),
        "next_state_cate_fea": tf.placeholder(dtype=tf.int64, shape=[None, low_cate_num], name='next_state_cate_fea'),
        "cur_state_dynamic_fea": tf.placeholder(dtype=tf.float32, shape=[None, low_dynamic_num+21], name='cur_state_fea'),
        "next_state_dynamic_fea": tf.placeholder(dtype=tf.float32, shape=[None, low_dynamic_num+21], name='next_state_fea'),
 
    }
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    return serving_input_receiver_fn


def estimator_save(estimator, params, log_dir):
    """ demo: estimator save """
    # save saved model
    serving_input_receiver_fn = export_serving_model_input(params)
    serving_model_path = log_dir
    logger.info("serving_model_path: {}".format(serving_model_path))
    estimator.export_savedmodel(serving_model_path, serving_input_receiver_fn=serving_input_receiver_fn)


def export_model_info(params):
    return params


def save_nn_model_info(params, model_info_file):
    model_info = export_model_info(params)
    json_data = json.dumps(model_info)
    fout = tf.gfile.Open(model_info_file, "w")
    fout.write(json_data)
    fout.close()


def custom_estimator(params, config):

    if params['task_type'] in 'low':
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=
                    params['high_model_dir'],# 或者hdfs路径 
            vars_to_warm_start=
                    ['high_controller/.*','model/variable/embed_table/.*'])
    else:
        ws=None

    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=config,
        warm_start_from=ws
    )

def get_constraint_loss(target, pred, le_loss=False):
    if le_loss:
        return tf.where(tf.less_equal(target, pred), tf.square(target - pred), 0.0)
    return tf.square(target - pred)


def update_lambda_vector_multi_step_no_dependency(config, lambda_vector, qnet_logits, total_budgets, target_real, use_bcq):
    # learning rate
    auto_lambda_alpha = [float(s) for s in config['lambda_alpha'].split("_")]
    auto_lambda_alpha = [tf.reshape(tf.convert_to_tensor(x), [1]) for x in auto_lambda_alpha]
    auto_lambda_alpha = tf.concat(auto_lambda_alpha, axis=0)
    op = tf.no_op()

    def _update_lambda_vector_once(lambda_vector_):
  
        calibration_vector = get_calibration_vector(lambda_vector_)
      
        qnet_logits_local, imt, i = qnet_logits
        actions_real,_ = select_action(qnet_logits_local, imt, i,use_bcq,calibration_vector,True)
        #actions_real = tf.map_fn(select_action,(qnet_logits_local, imt, i,use_bcq,calibration_vector),name='argmax')
        budgets = action_to_budgets(actions_real,total_budgets) # [B,4]

        mean_real = tf.reduce_mean(budgets-target_real,axis=0)
        
        delta_lambda = auto_lambda_alpha*(mean_real/target_real)

        delta_lambda = tf.Print(delta_lambda, [delta_lambda], "#delta_lambda=========", summarize=10)
        lambda_vector_ = tf.Print(lambda_vector_, [lambda_vector_], "#lambda_vector_", summarize=10)
        return lambda_vector_ + delta_lambda
    # multi_step update
    # qnet_logits = tf.stop_gradient(qnet_logits)
    iter_lambda_vector = lambda_vector
    for _ in range(int(config["lambda_update_num"])):
        iter_lambda_vector = _update_lambda_vector_once(iter_lambda_vector)

    op = tf.assign(lambda_vector, iter_lambda_vector)
    update_op = tf.group([op])
    return update_op


def approx_argmax_(x, epsilon=1e-10,approx_beta_rate=40.0):
    #  x -> [B,20]
    data_len = 20
    beta = approx_beta_rate # [1]
    norm_x = x / (abs(tf.reduce_max(x,axis=1,keepdims=True)) + epsilon) # [B,20]
    exp_x = tf.exp(beta * norm_x) # [B,20]
    cost = tf.reshape(tf.range(0,tf.cast(data_len, tf.float32)),[1,data_len]) # [20]
    exp_x_sum =  tf.reduce_sum(exp_x,axis=1,keepdims=True) #[B,1]
    return tf.reduce_sum(exp_x*cost/exp_x_sum,axis=1,keepdims=True) # [B,1]


def get_approx_action(qnet_logits, imt, i, use_bcq, threshold):
    if use_bcq:
        imt = tf.exp(imt)
        imt = (imt / tf.reduce_max(imt) > threshold)
        imt = tf.cast(imt, dtype=tf.float32)
        phase_qnet_logits = imt * qnet_logits + (1. - imt) * -1e8
    else:
        phase_qnet_logits = qnet_logits  
    approx_action = approx_argmax_(phase_qnet_logits)
    return approx_action


def z_score_norm(x, epsilon=1e-10):
    """
    z-score normalization
    :param x: input 1D Tensor
    :return: output 1D Tensor
    """
    mu = tf.reduce_mean(x)
    sigma = tf.sqrt(tf.reduce_mean(tf.square(x - mu)))
    return (x - tf.reduce_mean(x))/(sigma + epsilon)

def action_to_budgets(action,total_budgets,action_dim=20,prod_num=4):
    # [B,4], [B,4]
    logger.info("action:{},total_budgets:{}".format(action.get_shape(),total_budgets.get_shape()))
    percentage = tf.multiply(tf.cast(action,tf.float32),tf.constant(1./action_dim,dtype=tf.float32))
    real_budgets = percentage*total_budgets

    return real_budgets

def get_calibration_vector(predict_lambda=None):
    action_dim = 20
    if predict_lambda is None:
        predict_lambda = [0.0, 0.0, 0.0, 0.0]
    vector = []
    for phase_index in range(4):
        temp_vector = []
        for index in range(action_dim):
            temp_vector.append(predict_lambda[phase_index]*index/action_dim)
        vector.append(temp_vector)
    return vector

def get_normalized_feature(feature,mean,variance,epsilon = 0.0000000001):
    return (feature - mean) / (tf.sqrt(variance) + epsilon)

def get_hash_cate_feature(feature,prefix,embedding_matrix,cate_fea_num,vocab_size,embed_dim,prod_num):
    logger.info("prefix:{},featur:{}".format(prefix.get_shape(),feature.get_shape()))
    label = feature
    feature_col=tf.map_fn(lambda x: tf.string_join([prefix,x],separator='_'),label)
    feature_hash = tf.string_to_hash_bucket_strong(feature_col, vocab_size, [1005, 1070])
    embed_cate_fea_col = tf.nn.embedding_lookup_hashtable_v2(embedding_matrix, feature_hash)
    reshaped_embed_cate_fea_col = tf.reshape(embed_cate_fea_col, [-1, cate_fea_num * embed_dim, prod_num])
    logger.info("reshaped_embed_cate_fea_col{}".format(reshaped_embed_cate_fea_col.get_shape()))
    return reshaped_embed_cate_fea_col
    

class LambdaHelper():

    def __init__(self,click_model,cost_model,save_prod):
        self.click_model = click_model
        self.cost_model = cost_model
        self.coefficient = 10
        self.save_prod = save_prod


    def get_optimal_lambda(self,pack_state):
        state,budgets = pack_state 
        #state [s_dim+1,4])
        beta = 10 
        # state [S] budgets[1]
        sample_size = 10
        lambda_min = 0
        lambda_max = sample_size
        budgets =  tf.tile(tf.reshape(budgets,[1,1]),[sample_size,1])/self.coefficient # [sample,1]
        
        lambda_set = tf.reshape(tf.range(lambda_min,lambda_max,1),[sample_size,1]) # sample,1
        lambda_set = tf.cast(lambda_set/sample_size,tf.float32) 

        state =tf.tile(tf.expand_dims(state,axis=0),[sample_size,1]) # 10,S
        new_s = tf.concat([state,lambda_set],axis=1) # 10,S+1

        _,cost,_,_,_  =self.cost_model.forward(new_s,None,'main',None,save_prod=self.save_prod) # [10,1]
        #_,reward,_,_,_   = self.click_model.forward(new_s,None,'main',None,save_prod=self.save_prod)# [10,1,4]

        cost =  tf.reshape(cost,[sample_size,1])
        #reward = tf.reshape(reward,[sample_size,1])
        # less_condition = tf.less(cost,budgets/beta)
        # zeros = tf.zeros_like(reward,dtype=tf.float32)
        # reward = tf.where(less_condition,reward,zeros)
        #lambda_argmax = tf.argmax(reward,axis=1) # [B,1,4]
        objective = tf.abs(cost-budgets)

        lambda_argmin = tf.cast(tf.argmin(objective,axis=0)/sample_size,tf.float32)

        return tf.reshape(lambda_argmin,[1])



if __name__=='__main__':
    x= tf.ones((64,20),dtype=tf.float32)
    approx_argmax_(x)
    print(x)
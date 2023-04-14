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

def load_normalization_parameter(mean_var_filename, fea_num, prod_num=4,use_bcorle=False):
    fea_mean = []
    fea_var = []
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
        if use_bcorle:
            fea_mean.append(1.0)
            fea_var.append(1.0)
        break
    logger.info('num_fea_mean %s', fea_mean)
    logger.info('num_fea_var %s', fea_var)
    logger.info("mean_var_filename%s",mean_var_filename)
   
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
        poi_id = features['poi_id']

        cur_state_cate_fea_col = features['cur_state_cate_fea']
        next_state_cate_fea_col = features['next_state_cate_fea']
        cur_state_dynamic_fea_col = features['cur_state_dynamic_fea']
        next_state_dynamic_fea_col = features['next_state_dynamic_fea']

        reward = tf.cast(features['reward'], tf.float32)
        action_col = tf.cast(features['action'],tf.int64)


        with tf.name_scope('variable'):
            embedding_matrix = tf.get_variable(
                            'embeddings',
                            shape=[vocab_size, embed_dim],
                            trainable=True,
                        )
     

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

                high_cate_feature = tf.tile(high_cate_feature,[1,1,4])
                high_next_cate_feature = tf.tile(high_next_cate_feature,[1,1,4])
                low_cate_feature = tf.tile(low_cate_feature,[1,1,4])
                low_next_cate_feature = tf.tile(low_next_cate_feature,[1,1,4])

                cur_state_dynamic_fea_col = tf.tile(tf.expand_dims(cur_state_dynamic_fea_col,axis=-1),[1,1,4])
                next_state_dynamic_fea_col = tf.tile(tf.expand_dims(next_state_dynamic_fea_col,axis=-1),[1,1,4])

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

        if (mode == tf.estimator.ModeKeys.PREDICT or ext_is_predict_serving or not use_bcorle) and not task_type in 'high':
            lambda_vector=tf.tile(tf.reshape(tf.ones_like(features['prod'],dtype=tf.float32),[-1,1,1]),[1,1,4])
            cur_state_dynamic_fea_col = tf.concat([cur_state_dynamic_fea_col,lambda_vector],axis=1)
            next_state_dynamic_fea_col = tf.concat([next_state_dynamic_fea_col,lambda_vector],axis=1)
            use_bcorle = True

        global_step = tf.train.get_or_create_global_step()
        
        with tf.name_scope('dict'):
            high_state_dynamic_mean_vec, high_state_dynamic_var_vec = load_normalization_parameter(
                high_state_dynamic_fea_mean_var_filename,
                high_dynamic_num,
                prod_num,
                use_bcorle=use_bcorle
            )
            if task_type in 'low':
                low_state_dynamic_mean_vec, low_state_dynamic_var_vec = load_normalization_parameter(
                    low_state_dynamic_fea_mean_var_filename,
                    low_dynamic_num,
                    prod_num,
                    use_bcorle=use_bcorle
                )

        if use_adaptive and task_type in 'high':
            lambda_vector_high = get_calibration_vector(auto_lambda_vector) 
        else:
            lambda_vector_high = [None,None,None,None]
        if task_type in 'low':
            lambda_vector_high = [None,None,None,None] # TODO 每个预算对应一个lambda
            lambda_vector_low = [None,None,None,None] # lambda泛化方案求解lambda
        

        high_model = MAModel(deep_layers=deep_layers,num_action=num_action,optimizer='adam',variable_scope='high_controller',learning_rate=learning_rate,
                                use_bn=use_bn,use_bcq=use_bcq,threshold=threshold,use_rem=use_rem,num_heads=num_heads,logger=logger)

        low_model = MAModel(deep_layers=deep_layers,num_action=num_action,optimizer='adam',variable_scope='low_controller',learning_rate=learning_rate,
                                use_bn=use_bn,use_bcq=use_bcq,threshold=threshold,use_rem=use_rem,num_heads=num_heads,logger=logger)
        
        if task_type in 'high':

            total_budgets = tf.cast(features['total_budgets'],tf.float32)
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
                    target = [float(s) for s in params["constraint_target"].split("_")]
                    constraint_weight = [float(s) for s in params["constraint_loss_weight"].split("_")]
                
                    dj_actions = get_approx_action(total_q_logits[:,0,:],total_q_imts[:,0,:],total_q_i[:,0,:],use_bcq,threshold)
                    bj_actions = get_approx_action(total_q_logits[:,1,:],total_q_imts[:,1,:],total_q_i[:,1,:],use_bcq,threshold)
                    ss_actions = get_approx_action(total_q_logits[:,2,:],total_q_imts[:,2,:],total_q_i[:,2,:],use_bcq,threshold)
                    push_actions =get_approx_action(total_q_logits[:,3,:],total_q_imts[:,3,:],total_q_i[:,3,:],use_bcq,threshold)
                    
                    actions_all = tf.concat([dj_actions,bj_actions,ss_actions,push_actions],axis=1)

                    actions_cpc = reward / total_budgets *actions_all
                    # cpc = click/cost 
                    dj_actions_mean = tf.reduce_mean(actions_cpc[:,0])
                    bj_actions_mean = tf.reduce_mean(actions_cpc[:,1])
                    ss_actions_mean = tf.reduce_mean(actions_cpc[:,2])
                    push_actions_mean = tf.reduce_mean(actions_cpc[:,3])
                    
                    dj_loss = get_constraint_loss(target=target[0], pred=dj_actions_mean)
                    bj_loss = get_constraint_loss(target=target[1], pred=bj_actions_mean)
                    ss_loss = get_constraint_loss(target=target[2], pred=ss_actions_mean)
                    push_loss = get_constraint_loss(target=target[3], pred=push_actions_mean)

                    all_constraint_loss = dj_loss*constraint_weight[0]+bj_loss*constraint_weight[1]+ss_loss*constraint_weight[2]+push_loss*constraint_weight[3]
            
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

            current_state = tf.concat([low_cate_feature,cur_state_dynamic_fea_col],axis=1)
            next_state = tf.concat([low_next_cate_feature,next_state_dynamic_fea],axis=1)
            
            is_terminal = tf.tile(tf.cast(features['is_terminal'], tf.float32),[1,prod_num])

            higher_actions,_,_,_,_ = high_model.forward(tf.concat([high_cate_feature,cur_state_dynamic_fea_col[:,:high_dynamic_num,:]],axis=1),lambda_vector_high,'main')
            # higher_actions = tf.reshape(tf.as_string(higher_actions),[-1,1,4])
            # budget_prefix = tf.constant([[str(i)+"_budget" for _ in range(4)]]) 
            # higher_budgets = get_hash_cate_feature(higher_actions,budget_prefix,embedding_matrix,1,vocab_size,embed_dim,prod_num)
            
            higher_budgets = tf.cast(tf.reshape(higher_actions,[-1,1,4]),tf.float32)
            higher_budgets = tf.concat([current_state,higher_budgets],axis=1)
            next_state = tf.concat([next_state,higher_budgets],axis=1)
            
            logger.info("current_state{}".format(current_state.get_shape()))
            best_actions,best_action_q,total_q_logits,total_q_imts,total_q_i = low_model.forward(current_state,lambda_vector_low,'main')
            _,next_best_action_q,_,_,_= low_model.forward(next_state,lambda_vector_low,'target')
            
            prod = tf.cast(features['prod'] , tf.int64)
            prod = tf.subtract(prod, 1)
    
            if mode == tf.estimator.ModeKeys.PREDICT:
                prod = tf.cast(tf.reshape(prod,[-1]),tf.int64)
                prod_one_hot = tf.one_hot(indices=prod, depth=4)

                best_actions = tf.cast(tf.reshape(best_actions,[-1,4]), dtype=tf.float32)
                best_action_q = tf.reshape(best_action_q,[-1,4])

                best_actions = tf.reduce_sum(tf.multiply(prod_one_hot,best_actions),axis=1,keepdims=True)
                best_action_q = tf.reduce_sum(tf.multiply(prod_one_hot,best_action_q),axis=1,keepdims=True)
                logger.info("best_actions:{},best_action_q:{}".format(best_actions.get_shape(),best_action_q.get_shape()))
                if ext_is_predict_serving == 1:
                    tf.identity(tf.cast(best_actions, dtype=tf.float32))
                    result = tf.concat([tf.cast(best_actions,dtype=tf.float32),best_action_q], axis=1, name="output_action")
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=result)
                else:
                    predictions = {
                        "action": best_actions,
                        "qvalue": best_action_q,
                        "poi_id": poi_id,
                        "cur_action": action_col,
                        "pvid":features["pvid"]
                    }
                    logger.info("offline predict")
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

         
            h_index = tf.reshape(prod,[-1,1])
            line = tf.cast(tf.reshape(tf.range(tf.shape(prod)[0]),[-1,1]),tf.int64)
            index = tf.concat([line,h_index],axis = 1)


            target_q = reward + (1 - is_terminal) * gamma * next_best_action_q 
            selected_q = high_model.get_actions_qvalue(total_q_logits,action_col)

            with tf.name_scope('output'):
                logger.info("total_q_logits:{},total_q_imts:{},selected_q:{},target_q:{},best_actions:{},best_action_q:{}".format(
                    total_q_logits.get_shape(),total_q_imts.get_shape(),selected_q.get_shape(),target_q.get_shape(),best_actions.get_shape(),best_action_q.get_shape()
                ))
                
                action_col = tf.squeeze(tf.gather_nd(action_col,index))
                total_q_logits = tf.squeeze(tf.gather_nd(total_q_logits,index))
                total_q_imts = tf.squeeze(tf.gather_nd(total_q_imts,index))
                selected_q = tf.squeeze(tf.gather_nd(selected_q,index))
                target_q = tf.squeeze(tf.gather_nd(target_q,index))

                best_actions = tf.squeeze(tf.gather_nd(best_actions,index))
                best_action_q = tf.squeeze(tf.gather_nd(best_action_q,index))

                logger.info("total_q_logits:{},total_q_imts:{},selected_q:{},target_q:{},best_actions:{},best_action_q:{}".format(
                    total_q_logits.get_shape(),total_q_imts.get_shape(),selected_q.get_shape(),target_q.get_shape(),best_actions.get_shape(),best_action_q.get_shape()
                ))

    


        is_weights =tf.cast(tf.ones_like(action_col), tf.float32)  # 为后续增加权重做准备，当前是1

        error = tf.reduce_mean(tf.abs(target_q - selected_q)) #[256,4] vs. [256]
        q_loss = q_loss_weight * Smooth_L1_Loss(target_q, selected_q, "loss", is_weights)
        all_loss = q_loss
        
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
            all_loss = q_loss + i_loss + i_reg_loss
            logger.info('i_loss {}'.format(i_loss))
            logger.info('i_reg_loss {}'.format(i_reg_loss))
            tf.summary.scalar('i_loss', i_loss)
            tf.summary.scalar('i_reg_loss', i_reg_loss)

        if use_batch_loss and task_type in 'high':
            all_loss += all_constraint_loss
        
        logger.info('q_loss {}'.format(q_loss))
        logger.info('all_loss {}'.format(all_loss))
        
        main_qnet_var = []
        target_qnet_var = []
        for i in range(prod_num):
            main_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller/agent{}/main_{}_net'.format(task_type,i,i)))
            target_qnet_var.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}_controller/agent{}/target_{}_net'.format(task_type,i,i)))

        tf.summary.scalar('total_action_qvalue', tf.reduce_mean(total_q_logits))
        tf.summary.scalar('best_action_qvalue', tf.reduce_mean(best_action_q))
        tf.summary.scalar('reward', tf.reduce_mean(reward))
        tf.summary.scalar('loss', all_loss)
        tf.summary.scalar('q_loss', q_loss)

        tf.summary.scalar('abs_error', error)
        
        all_main_var = main_qnet_var[0]+main_qnet_var[1]+main_qnet_var[2]+main_qnet_var[3]
        all_target_var = target_qnet_var[0]+target_qnet_var[1]+target_qnet_var[2]+target_qnet_var[3]
        logger.info(all_main_var)
        logger.info(all_target_var)
      
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

        if task_type in 'high':
            train_op = high_model.get_train_op(global_step,all_loss)
            update_op.append(update_lambda_op)
            update_op.append(update_target_lambda_op)

        elif task_type in 'low': # todo hashtable reload
            high_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='high_controller')
            hash_map = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embeddings')
            logger.info("hash_map{}".format(hash_map))
            logger.info("high_vars{}".format(high_vars))
            all_vars = high_vars+hash_map
         
            assignment_map = {re.match("^(.*):\\d+$", var.name).group(1): var for var in all_vars}
            high_model_dir = params["high_model_dir"]
            logger.info("[high controller] assignment_map: {}".format(assignment_map))
            logger.info("high_model_dir: {}".format(high_model_dir))
            tf.train.init_from_checkpoint(high_model_dir, assignment_map)
            logger.info("load succ!")
            
            train_op = low_model.get_train_op(global_step,all_loss)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.control_dependencies(update_op):
                var_diff = tf.add_n(
                    [tf.reduce_mean(tf.squared_difference(t, e)) for t, e in zip(all_main_var, all_target_var)])
                tf.summary.scalar('var_diff', tf.reduce_mean(var_diff))
                # train_op = HashAdamOptimizer(learning_rate=learning_rate).minimize(all_loss,global_step=global_step)
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
    ops = [tf.assign_add(neg_counter, 1), ]
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
      
        "is_terminal": tf.placeholder(dtype=tf.float32, shape=[None, 1], name='is_terminal'),
        "reward": tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward'),
        "action": tf.placeholder(dtype=tf.int64, shape=[None, 1], name='action'),

        "cur_state_cate_fea": tf.placeholder(dtype=tf.int64, shape=[None, low_cate_num], name='cur_state_cate_fea'),
        "next_state_cate_fea": tf.placeholder(dtype=tf.int64, shape=[None, low_cate_num], name='next_state_cate_fea'),
        "cur_state_dynamic_fea": tf.placeholder(dtype=tf.float32, shape=[None, low_dynamic_num], name='cur_state_fea'),
        "next_state_dynamic_fea": tf.placeholder(dtype=tf.float32, shape=[None, low_dynamic_num], name='next_state_fea'),
 
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

    # if params['task_type'] in 'low':
    #     ws = tf.estimator.WarmStartSettings(
    #         ckpt_to_initialize_from=
    #                 params['high_model_dir'],# 或者hdfs路径 
    #         vars_to_warm_start=
    #                 ['high_controller/.*','model/variable/embed_table/.*'])
    # else:
    #     ws=None
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
    action_dim = tf.shape(x)[1]
    # x = x - tf.reduce_mean(x)
    cost = tf.range(0,tf.cast(action_dim, tf.float32))
    beta = float(approx_beta_rate)/(abs(tf.reduce_max(x)) + epsilon)
    exp_x = tf.exp(beta * x)
    cost = exp_x*cost # [B,20]

    return tf.reduce_sum(cost/ tf.reduce_sum(exp_x,axis=1,keepdims=True),axis=1,keepdims=True)


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
    embed_cate_fea_col = tf.nn.embedding_lookup(embedding_matrix, feature_hash)
    reshaped_embed_cate_fea_col = tf.reshape(embed_cate_fea_col, [-1, cate_fea_num * embed_dim, prod_num])
    logger.info("reshaped_embed_cate_fea_col{}".format(reshaped_embed_cate_fea_col.get_shape()))
    return reshaped_embed_cate_fea_col
    

if __name__=='__main__':
    x= tf.ones((64,20),dtype=tf.float32)
    approx_argmax_(x)
    print(x)
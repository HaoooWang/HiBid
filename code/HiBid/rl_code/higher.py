#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 2)[0]  # 上2级目录
sys.path.append(config_path)


class MAModel:
    def __init__(self,deep_layers,num_action,optimizer,variable_scope,learning_rate,use_bn,use_bcq,threshold,use_rem,num_heads,logger,use_mask=False):
        self.deep_layers = deep_layers
        self.num_action = num_action
        self.use_bn = use_bn
        self.use_rem = use_rem
        self.num_heads = num_heads
        self.logger = logger

        self.use_bcq = use_bcq
        self.threshold = threshold
        self.use_action_mask = use_mask

        self.optimizer = optimizer 
        self.variable_scope = variable_scope
        self.learning_rate = learning_rate
        
    
    def forward(self,states,lambda_vector,name='main',mask=None,save_prod=None):
        if lambda_vector is None: lambda_vector=[None,None,None,None]
        if mask is None:
            new_mask = [None,None,None,None]
        else:
            new_mask = [mask[:,0,:],mask[:,1,:],mask[:,2,:],mask[:,3,:]]
        total_q_logits, total_q_imts, total_q_i= [], [], []
        best_actions, best_action_q = [],[]
        trainable=True if name=='main' else False
        if save_prod is not None:
            with tf.variable_scope('{}/agent{}'.format(self.variable_scope,save_prod),reuse=tf.AUTO_REUSE):
                
                random_coeff = tf.constant(make_coeff(self.num_heads), tf.float32)
                q_logits, q_imt, q_i = BCQ_net("{}_{}".format(name,save_prod), states, self.deep_layers, self.num_action, self.use_bn, self.use_rem,
                                                            self.num_heads, random_coeff , trainable, self.logger)
                q_best_action , best_action_qvalue = self.select_actions(q_logits , q_imt , lambda_vector[save_prod],mask=new_mask[save_prod])

                return q_best_action,best_action_qvalue,q_logits,q_imt,q_i

        for i in range(4):
            with tf.variable_scope('{}/agent{}'.format(self.variable_scope,i),reuse=tf.AUTO_REUSE):
               
                random_coeff = tf.constant(make_coeff(self.num_heads), tf.float32)
                q_logits, q_imt, q_i = BCQ_net("{}_{}".format(name,i), states[:, :, i], self.deep_layers, self.num_action, self.use_bn, self.use_rem,
                                                            self.num_heads, random_coeff , trainable, self.logger)

                q_logits = tf.reshape(q_logits,[-1,20])
                total_q_imts.append(q_imt)
                total_q_i.append(q_i)
                total_q_logits.append(q_logits)

                q_best_action , best_action_qvalue = self.select_actions(q_logits , q_imt , lambda_vector[i],mask=new_mask[i])
                
                best_actions.append(q_best_action)
                best_action_q.append(best_action_qvalue)

        total_q_logits =tf.stack(total_q_logits, axis=1)
        total_q_imts =tf.stack(total_q_imts, axis=1)
        total_q_i =tf.stack(total_q_i, axis=1)

        best_actions =tf.squeeze(tf.stack(best_actions, axis=1))
        best_action_q =tf.squeeze(tf.stack(best_action_q, axis=1))

        self.logger.info("total_q_logits:{},total_q_imts:{},total_q_i:{},best_actions:{},best_action_q:{}".format(total_q_logits.get_shape(),total_q_imts.get_shape(),total_q_i.get_shape(),best_actions.get_shape(),best_action_q.get_shape()))
        return best_actions,best_action_q,total_q_logits,total_q_imts,total_q_i
    
    def get_actions_qvalue(self,q_logits,actions):
        # q_logits [B,4,20] actions [B,4]
        one_hot_action = tf.one_hot(indices=actions, depth=self.num_action, dtype=tf.float32)
        qvalue = tf.reduce_sum(tf.multiply(q_logits , one_hot_action), axis=2)
        return qvalue

    def select_actions(self,q,imt,lambda_vector=None,batch=False,mask=None):
        q = tf.identity(q,name='extra_q')
        if self.use_action_mask and mask is not None:
            zeros = tf.zeros_like(q,dtype=tf.float32)
            new_q = tf.where(mask,q,zeros) #TODO 用argmin
        else:
            new_q = q 

        return_shape = [-1,4] if batch else [-1,1]
        argmax_axis = 2 if batch else 1
        if lambda_vector is not None:
            new_q  = q - lambda_vector
        else:
            new_q = new_q
        if self.use_bcq:
            imt = tf.exp(imt)
            imt = (imt / tf.reduce_max(imt, axis=1, keep_dims=True) >= self.threshold)
            imt = tf.cast(imt, dtype=tf.float32)
            return tf.reshape(tf.argmax(imt * new_q + (1. - imt) * -1e8, axis=argmax_axis), return_shape), tf.reshape(
                tf.reduce_max(imt * new_q + (1. - imt) * -1e8, axis=argmax_axis), return_shape)
        else:
            return tf.reshape(tf.argmax(new_q, axis=argmax_axis), return_shape), tf.reshape(
                tf.reduce_max(new_q, axis=argmax_axis), return_shape) 

    def get_train_op(self, global_step, loss):
        optimizer = get_optimizer_by_name(self.optimizer)(self.learning_rate)
        trainable_vars = []
        all_vars = [var for var in tf.trainable_variables()]
        for var in all_vars:
            if self.variable_scope in var.name:
                trainable_vars.append(var)
        # return a list of trainable variable in you model
        # params = tf.trainable_variables()
        # create an optimizer
        # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # compute gradients for params
        gradients = tf.gradients(loss, trainable_vars)
        # process gradients
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5)

        train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_vars), global_step=global_step)
        # train_op = optimizer.minimize(loss, global_step=global_step, var_list=trainable_vars)
        self.logger.info("[{}] trainable_vars: {}".format(self.variable_scope,trainable_vars))
        self.logger.info("[{}] all_vars: {}".format(self.variable_scope,all_vars))
        return train_op

    


def make_coeff(num_heads):
    arr = np.random.uniform(low=0.0, high=1.0, size=num_heads)
    arr /= np.sum(arr)
    return arr



optimizer_mapping = {
    "adam": tf.train.AdamOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "rmsprop": lambda lr: tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.95)
}


def get_optimizer_by_name(name):
    if name not in optimizer_mapping.keys():
        return optimizer_mapping["sgd"]
    return optimizer_mapping[name]
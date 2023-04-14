#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from util import *
import numpy as np

def BCQ_net(name, state_, hidden_units_list, num_actions, use_bn, use_rem, num_heads, random_coeff, trainable, logger):
    print("Load new file")
    #state_ = tf.Print(state_, [state_], "#state_", summarize=20)
    with tf.variable_scope(name + '_net'):
        net = state_
        if use_bn:
            net = tf.layers.batch_normalization(state_, axis=-1, momentum=0.99,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(),
                                                beta_regularizer=None,
                                                gamma_regularizer=None,
                                                beta_constraint=None,
                                                gamma_constraint=None,
                                                training=False,
                                                trainable=trainable,
                                                reuse=None,
                                                renorm=False,
                                                renorm_clipping=None,
                                                renorm_momentum=0.99,
                                                fused=None,
                                                name="bn"
                                                )
        with tf.variable_scope(name + 'q_net'):
            q_net = net
            for i in range(len(hidden_units_list)):
                q_net = tf.layers.dense(
                    inputs=q_net,
                    activation=tf.nn.relu,
                    units=hidden_units_list[i],
                    kernel_initializer=ortho_init(),
                    trainable=trainable,
                    name='fc_{idx}'.format(idx=i)
                )

            if use_rem:
                q = tf.layers.dense(q_net, num_actions * num_heads, activation=None, name="head_1", trainable=trainable)
                q = tf.reshape(q, (tf.shape(q_net)[0], num_actions, num_heads), name='head_2')
                q = tf.squeeze(tf.reduce_sum(q * tf.reshape(random_coeff, [1, 1, -1]), axis=2), name="head_q")
            else:
                q = tf.layers.dense(q_net, num_actions, activation=None, name="head_q", trainable=trainable)

        with tf.variable_scope(name + 'i_net'):
            # I network
            i_net = net
            for i in range(len(hidden_units_list)):
                i_net = tf.layers.dense(
                    inputs=i_net,
                    activation=tf.nn.relu,
                    units=hidden_units_list[i],
                    kernel_initializer=ortho_init(),
                    trainable=trainable,
                    name='fc_{idx}'.format(idx=i)
                )
            i = tf.layers.dense(i_net, num_actions, activation=None, name="i_o", trainable=trainable)
        imt = tf.nn.log_softmax(i)
    return q, imt, i

def ortho_init(scale=1.0):
    # belsopenai baselines 
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init

def MMOE_model(name, state_, hidden_units_list, num_actions, use_bn, use_rem, num_heads, random_coeff, trainable, logger):

	cur_state, cur_prodtype = state_[:, :-1], state_[:, -1]
	cur_prodtype = tf.Print(cur_prodtype, [cur_prodtype], "#cur_prodtype", summarize=10)
	cur_prodtype = tf.clip_by_value(tf.cast(tf.subtract(cur_prodtype, 1), tf.int64), 0, 3)
	r = tf.sqrt(tf.cast(6 / 8, dtype=tf.float32))
	base_emb_weights = tf.get_variable(name + '_prod_embeddings',
                                        shape=[5, 8],
                                        trainable=trainable,
                                        initializer=tf.random_uniform_initializer(minval=-r, maxval=r))
	cur_prodtype_state = tf.reshape(tf.nn.embedding_lookup(base_emb_weights, cur_prodtype), [-1, 8])
	with tf.variable_scope(name + '_net'):
		net = cur_state
		if use_bn:
			net = tf.layers.batch_normalization(cur_state, axis=-1, momentum=0.99,
			                                    epsilon=0.001,
			                                    center=True,
			                                    scale=True,
			                                    beta_initializer=tf.zeros_initializer(),
			                                    gamma_initializer=tf.ones_initializer(),
			                                    moving_mean_initializer=tf.zeros_initializer(),
			                                    moving_variance_initializer=tf.ones_initializer(),
			                                    beta_regularizer=None,
			                                    gamma_regularizer=None,
			                                    beta_constraint=None,
			                                    gamma_constraint=None,
			                                    training=False,
			                                    trainable=trainable,
			                                    reuse=None,
			                                    renorm=False,
			                                    renorm_clipping=None,
			                                    renorm_momentum=0.99,
			                                    fused=None,
			                                    name="bn"
			                                    )
		with tf.variable_scope(name + 'q_net'):
			# q_net = net
			# for i in range(len(hidden_units_list)):
			#     q_net = tf.layers.dense(
			#         inputs=q_net,
			#         activation=tf.nn.relu,
			#         units=hidden_units_list[i],
			#         kernel_initializer=ortho_init(),
			#         trainable=trainable,
			#         name='fc_{idx}'.format(idx=i)
			#     )
			q = mmoe_net(net, num_actions, num_heads, cur_prodtype_state, cur_prodtype, trainable, logger)

			## udm or dense
			q = head_net(q, use_rem, num_actions, num_heads, random_coeff, 0, trainable)

		with tf.variable_scope(name + 'i_net'):
			# i_net = net
			# for i in range(len(hidden_units_list)):
			# 	i_net = tf.layers.dense(
			# 		inputs=i_net,
			# 		activation=tf.nn.relu,
			# 		units=hidden_units_list[i],
			# 		kernel_initializer=ortho_init(),
			# 		trainable=trainable,
			# 		name='fc_{idx}'.format(idx=i)
			# 	)
			i_net = mmoe_net(net, num_actions, num_heads, cur_prodtype_state, cur_prodtype, trainable, logger)
			i = tf.layers.dense(i_net, num_actions, activation=None, name="i_o", trainable=trainable)
		imt = tf.nn.log_softmax(i)
	return q, imt, i

def mmoe_net(cur_state, num_actions, num_heads, cur_prodtype_state, cur_prodtype, trainable, logger):
	## MMOE
	gates_num = 4
	experts_num = 4
	experts_hidden_unit_list = map(int, '64_32'.split('_'))  # [1024,512]
	gates_hidden_unit_list = map(int, '32'.split('_'))  # [512,128,32]

	## expert net
	# [(3000, 512, 1), (3000, 512, 1)]  experts_num = 2
	expert_list = [experts_net(cur_state, experts_hidden_unit_list, 'ALL_expert_%d' % i, trainable)
	               for i in range(experts_num)]
	# (3000, 512, 2)
	concat_expert = tf.concat(expert_list, axis=2, name='concat_expert_layer')

	## gate net
	# [(3000, 1, 2), (3000, 1, 2)] gates_num = 2
	gate_list = [gates_net(cur_prodtype_state, gates_hidden_unit_list, experts_hidden_unit_list[-1], experts_num,
	                       'Task%d_gate' % (i + 1), trainable)
	             for i in range(gates_num)]

	## output calc
	# [(3000, 512, 2), (3000, 512, 2)]
	weighted_expert_list = [tf.multiply(concat_expert, gate)
	                        for gate in
	                        gate_list]

	# [(3000, 1, 512), (3000, 1, 512)]
	weighted_expert_out_list = [tf.reshape(tf.reduce_sum(weighted_expert, axis=2), [-1, 1, experts_hidden_unit_list[-1]])
	                            for weighted_expert in
	                            weighted_expert_list]

	# (3000, 4, 512)
	weighted_expert_out_list = tf.concat(weighted_expert_out_list, axis=1)

	# (3000, 4, 1)
	cur_prodtype = tf.cast(tf.reshape(tf.one_hot(cur_prodtype, gates_num), [-1, gates_num, 1]), tf.float32)

	# (3000, 1, 512)
	weighted_expert_out_list = tf.reduce_sum(tf.multiply(weighted_expert_out_list, cur_prodtype), axis=1)

	# (3000, 512)
	weighted_expert_out_list = tf.reshape(weighted_expert_out_list, [-1, experts_hidden_unit_list[-1]])

	return weighted_expert_out_list

def experts_net(input_layer, hidden_units_list, name, trainable):
	with tf.name_scope(name):
		# kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False) ## CLEAN
		net = input_layer
		for i in range(len(hidden_units_list)):
			net = tf.layers.dense(inputs=net,
                                units=hidden_units_list[i],
                                activation=None,
                                trainable=trainable,
                                kernel_initializer=tf.glorot_normal_initializer(),
                                bias_initializer=tf.glorot_normal_initializer(),
                                name='%s_fc_%d' % (name, i))
			net = mish(net)
	return tf.reshape(net, (-1, hidden_units_list[-1], 1))

def gates_net(input_layer, hidden_units_list, expert_unit, output_dim, name, trainable):
	with tf.name_scope(name):
		net = input_layer
		net = tf.layers.dense(inputs=net,
                            units=output_dim,
                            activation=tf.nn.softmax,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            bias_initializer=tf.glorot_normal_initializer(),
                            trainable=trainable,
                            name='%s_softmax_output' % (name))
	return tf.reshape(net, [-1, 1, output_dim])


def head_net(q_net, use_rem, num_actions, num_heads, random_coeff, i, trainable):
	if use_rem:
		q = tf.layers.dense(q_net, num_actions * num_heads, activation=None, name='head_1_{idx}'.format(idx=i),
		                    trainable=trainable)
		q = tf.reshape(q, (tf.shape(q_net)[0], num_actions, num_heads), name='head_2_{idx}'.format(idx=i))
		q = tf.squeeze(tf.reduce_sum(q * tf.reshape(random_coeff, [1, 1, -1]), axis=2),
		                name='head_q_{idx}'.format(idx=i))
	else:
		q = tf.layers.dense(q_net, num_actions, activation=None, name='head_q_{idx}'.format(idx=i), trainable=trainable)
	return q

def mish(x):
	return tf.multiply(x, tf.tanh(tf.math.softplus(x)))

def Qmix(state_, q_vals_low, embed_dim_low, agent_num, trainable):
    with tf.variable_scope('mix_w1'):
        w1 = tf.layers.dense(
            inputs=state_,
            activation=tf.nn.relu,
            units=embed_dim_low * agent_num,
            kernel_initializer=ortho_init(),
            trainable=trainable,
            name='qmix_w1'
        )
    with tf.variable_scope('mix_w1'):
        w2 = tf.layers.dense(
            inputs=state_,
            activation=tf.nn.relu,
            units=embed_dim_low,
            kernel_initializer=ortho_init(),
            trainable=trainable,
            name='qmix_w2'
        )
    k = tf.matmul(w1, w2)
    k = k / tf.reduce_sum(k, axis=1, keep_dims=True)
    q_total = tf.matmul(q_vals_low, k)
    return tf.reshape(q_total, shape=[q_total.get_shape()[0], -1, 1])
    

def Mix_net(states, state_fea_num, trainable):
	hyper_w_1 = tf.layers.dense(
                    inputs=states,
                    activation=tf.nn.relu,
                    units=state_fea_num,
                    kernel_initializer=None,
                    trainable=trainable,
                    name='hyper_w_1'
                )
	hyper_w_final = tf.layers.dense(
                    inputs=states,
                    activation=tf.nn.relu,
                    units=state_fea_num,
                    kernel_initializer=None,
                    trainable=trainable,
                    name='hyper_w_final'
                )
	hyper_b_1 = tf.layers.dense(
                    inputs=states,
                    activation=tf.nn.relu,
                    units=state_fea_num,
                    kernel_initializer=None,
                    trainable=trainable,
                    name='hyper_b_1'
                )
	

	bs = states.size(0)
	w1 = tf.abs(hyper_w_1(states))
	w_final = tf.abs(hyper_w_final(states))
	w1 = tf.reshape(w1, [-1, 4, state_fea_num])
	w_final = w_final.view(-1, state_fea_num, 1)
	k = tf.reshape(tf.matmul(w1,w_final), [bs, -1, 4])
	k = k / tf.reduce_sum(k, dim=2, keepdim=True)

	b1 = tf.reshape(hyper_b_1(states), [-1, 1, state_fea_num])
	b = tf.matmul(b1, w_final) #  [-1, 1, state_fea_num] x -1, state_fea_num, 1)
	return k, b
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import traceback
import os

from model import *
from util.util import *
from data_generator import *

flags = tf.app.flags
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "chief", "One of 'ps', 'worker', 'chief', 'evaluator'")
flags.DEFINE_string("chief_hosts", "", "")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_string("task", 'train', "task: train, evaluate or test")


flags.DEFINE_integer("is_dist", '0', "is dist")
# flags.DEFINE_string("log_dir", "./log_dir", "log directory")
flags.DEFINE_string("model_filename", "dqn_model.json", "log directory")
flags.DEFINE_string("predict_result_path", "..", "log directory")
flags.DEFINE_string('cur_date', '20190000', 'cur_date')

flags.DEFINE_string("high_state_dynamic_fea_mean_var_filename", "", "mean and variance of cur state file")
flags.DEFINE_string("low_state_dynamic_fea_mean_var_filename", "", "mean and variance of next state file")

flags.DEFINE_string("high_state_dynamic_fea","","input state feature name list")
flags.DEFINE_string("high_state_cate_fea","","input state feature name list")
flags.DEFINE_string("low_state_dynamic_fea","","input state feature name list")
flags.DEFINE_string("low_state_cate_fea","","input state feature name list")

flags.DEFINE_integer("bit", 22, "bit")

flags.DEFINE_string("task_type",'high',"task_type: high or low")
flags.DEFINE_string("high_model_dir","","high controller model directory")
flags.DEFINE_integer("prod_num", 4, "prod num")

flags.DEFINE_bool("use_mask",False,"whether use action mask")
flags.DEFINE_bool("use_bcorle",False,"whether use bcorle")
flags.DEFINE_integer("bcorle_lambda_num",30,"number of exp reuse times")
flags.DEFINE_string("bcorle_lambda_range","0_1","lambda sample range")
flags.DEFINE_bool("use_adaptive",False,"whether use adaptive lambda")
flags.DEFINE_bool("use_batch_loss",False,"whether use batch loss")
flags.DEFINE_string("lambda_alpha","1e-1_1e-1_1e-1_1e-1","lambda_alpha")
flags.DEFINE_integer("lambda_update_num",10,"lambda_update_num")
flags.DEFINE_integer("lambda_update_interval",100,"lambda_update_interval")
flags.DEFINE_string("lambda_budgets_target","1_1_1_1","lambda_budgets_target")
flags.DEFINE_string("constraint_target","1_1_1_1","constraint_target")
flags.DEFINE_string("constraint_loss_weight","0.01_0.01_0.01_0.01","constraint_loss_weight")

flags.DEFINE_integer("batch_size", 512, "batch size")
flags.DEFINE_integer("epoch", 1, "epoch")
flags.DEFINE_string("optimizer", 'adam', "optimizer")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_string('deep_layers', '64', 'deep_layers')
flags.DEFINE_float('gamma', '0.95', 'gamma')
flags.DEFINE_integer('embed_dim', 18, 'embed_dim')
flags.DEFINE_integer("update_interval", 500, "update_interval")
flags.DEFINE_integer("use_ddqn", 0, "use_ddqn")
flags.DEFINE_integer("total_steps", 48, "total_steps")
flags.DEFINE_string("reward_alpha", "0.3,0.1", "reward_alpha")
flags.DEFINE_integer("save_summary_steps", 2000, "save summary step")
flags.DEFINE_integer("save_checkpoints_steps", 50000, "save checkpoints step")
flags.DEFINE_float("trainset_validset_ratio", 0.8, "train set ratio")
flags.DEFINE_integer("ext_is_predict_serving", 1, "predict op is　 for serving")
flags.DEFINE_bool("use_rem",True,"use_rem")
flags.DEFINE_bool("read_hive",True,"read_hive")
flags.DEFINE_bool("use_bcq",True,"use_bcq")
flags.DEFINE_bool("use_bn",False,"use_bn")
flags.DEFINE_bool("use_cate_fea", False, "use_cate_fea")
flags.DEFINE_bool("use_dense_hashtable",False,"use_dense_hashtable")
flags.DEFINE_float("i_loss_weight",1,"i_loss_weight")
flags.DEFINE_float("i_regularization_weight", 1e-2 ,"i_regularization_weight")
flags.DEFINE_float("q_loss_weight",1,"q_loss_weight")
flags.DEFINE_integer("num_heads",20,"num_heads")
flags.DEFINE_float("threshold",0.3,"threshold")
flags.DEFINE_integer("num_action",20,"num_action")
FLAGS = flags.FLAGS
logger = set_logger()

###############################
class Trainer(object):
    def __init__(self,data):
        self.init_environ()

        tf.disable_chief_training(shut_ratio=1.0, slow_worker_delay_ratio=10)
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            # device_filters=device_filters
        )
        session_config.gpu_options.allow_growth = True

        self.config = tf.estimator.RunConfig(
            save_summary_steps=FLAGS.save_summary_steps,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            model_dir=FLAGS.log_dir,
            session_config=session_config,
            keep_checkpoint_max=5
            # chief_training=False
        )
        logger.info("save_summary_steps:{}".format(FLAGS.save_summary_steps))
        logger.info("save_checkpoints_steps:{}".format(FLAGS.save_checkpoints_steps))

        self.params = {

            'high_state_dynamic_fea_mean_var_filename': FLAGS.high_state_dynamic_fea_mean_var_filename,
            'low_state_dynamic_fea_mean_var_filename': FLAGS.low_state_dynamic_fea_mean_var_filename,
            'learning_rate': FLAGS.learning_rate,
            'deep_layers': FLAGS.deep_layers,
            'vocab_size': 1 << FLAGS.bit,
            'gamma': FLAGS.gamma,
            'embed_dim': FLAGS.embed_dim,
            'update_interval': FLAGS.update_interval,
            'use_rem': FLAGS.use_rem,
            'use_bcq': FLAGS.use_bcq,
            'use_bn': FLAGS.use_bn,
            'use_dense_hashtable': FLAGS.use_dense_hashtable,
            'i_loss_weight': FLAGS.i_loss_weight,
            'i_regularization_weight': FLAGS.i_regularization_weight,
            'q_loss_weight': FLAGS.q_loss_weight,
            'num_heads': FLAGS.num_heads,
            'threshold': FLAGS.threshold,
            'total_steps': FLAGS.total_steps,
            'num_action': FLAGS.num_action,
            'ext_is_predict_serving': FLAGS.ext_is_predict_serving,
            'batch_size':FLAGS.batch_size,
            'use_cate_fea':FLAGS.use_cate_fea,

            "task":FLAGS.task,
            "task_type":FLAGS.task_type,
            "high_model_dir":FLAGS.high_model_dir,

            "prod_num":FLAGS.prod_num,
            'use_bcorle':FLAGS.use_bcorle,
            "use_adaptive":FLAGS.use_adaptive,
            "use_batch_loss":FLAGS.use_batch_loss,
            "use_mask":FLAGS.use_cpcas,

            "lambda_alpha":FLAGS.lambda_alpha,
            "lambda_update_interval":FLAGS.lambda_update_interval,
            "lambda_budgets_target":FLAGS.lambda_budgets_target,
            "lambda_update_num":FLAGS.lambda_update_num,
            "constraint_target":FLAGS.constraint_target,
            "constraint_loss_weight":FLAGS.constraint_loss_weight,

            "high_state_dynamic_fea":FLAGS.high_state_dynamic_fea,
            "high_state_cate_fea":FLAGS.high_state_cate_fea,
            "low_state_dynamic_fea":FLAGS.low_state_dynamic_fea,
            "low_state_cate_fea":FLAGS.low_state_cate_fea,
        }
        print(self.params)

        self.data = data 
        self.data.build_data_spec()

        self.estimator = custom_estimator(self.params, self.config)
        self.file_names = []

    def init_environ(self):
        if FLAGS.is_dist == 0:
            self.job_name = 'worker'
            self.task_index = 0
        else:
            self.job_name = FLAGS.job_name
            self.task_index = FLAGS.task_index
            self.ps_hosts = FLAGS.ps_hosts.split(",")
            self.worker_hosts = FLAGS.worker_hosts.split(",")
            self.chief_hosts = FLAGS.chief_hosts.split(",")
            logger.info('Chief host is :%s' % self.chief_hosts)
            logger.info('PS hosts are: %s' % self.ps_hosts)
            logger.info('Worker hosts are: %s' % self.worker_hosts)

            logger.info('job_name : %s' % self.job_name)
            logger.info('task_index : %d' % self.task_index)
            self.cluster = {'chief': self.chief_hosts, "ps": self.ps_hosts,
                            "worker": self.worker_hosts}
            logger.info('FLAGS.task: %s' % FLAGS.task)
            if(FLAGS.task=="train"):
                os.environ['TF_CONFIG'] = json.dumps(
                    {
                        'cluster': self.cluster,
                        'task': {
                            'type': self.job_name,
                            'index': self.task_index
                        }
                    }
                )
    
    def train(self):
        logger.info("FLAGS.read_hive:{}".format(FLAGS.read_hive))

        tf.disable_chief_training(shut_ratio=0.8, slow_worker_delay_ratio=1.2)
        tf.estimator.train_and_evaluate(self.estimator,self.data.train_spec, self.data.eval_spec)

    def predict(self):
        logger.info('job_name: %s', self.job_name)
        if self.job_name == 'chief':
            return

        line_num = 0
            
        predictions_iterator = self.estimator.predict(self.data.eval_input_fn, yield_single_examples=True)
        result_filename = FLAGS.predict_result_path+"/predict_result_"+str(FLAGS.task_index)+".csv"
        logger.info('predict_result_location %s', result_filename)
        fout_result_file = tf.gfile.Open(result_filename, 'w')
        logger.info('predict start')
        for prediction in predictions_iterator:
            logger.info('line_num: d%', line_num)
            line_num += 1
            if line_num % 50000 == 0:
                logger.info("write predict result %d lines\n", line_num)
                # logger.info("prediction: [pvid]")
                # logger.info(prediction['pvid'])
                logger.info("prediction: [poi_id]")
                logger.info(prediction['poi_id'])
                logger.info("prediction: [qvalue]")
                logger.info(prediction['qvalue'])
                logger.info("prediction: [action]")
                logger.info(prediction['action'])
                logger.info("prediction: [cur_action]")
                logger.info(prediction['cur_action'])
            line = "%s\t%s\t%s\t%s\t%s\t%s\n" % (
            str(prediction['pvid'][0]), 
            str(prediction['poi_id'][0]), str(prediction['qvalue'][0]), str(int(prediction['action'][0])),str(int(sum(prediction['cur_action']))),str(self.action_to_bid(prediction['prod'][0],prediction['action'][0])))
            fout_result_file.write(line)
        fout_result_file.close()
        logger.info('predict end')
        logger.info("total write predict result %d lines", line_num)


    def save_model_tfserving(self):
        if self.job_name == 'chief':
            self.params['ext_is_predict_serving'] = 1
            for i in range(4):
                self.params['save_prod'] = i
                self.estimator = custom_estimator(self.params, self.config)
                logger.info('model_dir %s', FLAGS.log_dir)
                logger.info("tfserving_model_dir=%s", FLAGS.log_dir + "/tfModel_{}".format(i))
                estimator_save(self.estimator, self.params, FLAGS.log_dir + "/tfModel_{}".format(i))
                save_nn_model_info(self.params, FLAGS.log_dir + "/" + "model_{}.json".format(i))
    
    def action_to_bid(self,prod,action):
        final_bid = 0.5+action*0.05
        return final_bid

def main(_):
    logger.info('FLAGS %s', FLAGS)
    logger.info('FLAGS.log_dir %s', FLAGS.log_dir)
    if FLAGS.job_name == "chief" and FLAGS.task_index == 0:
        if not tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.MakeDirs(FLAGS.log_dir)
    
    data_generator = DataGenerator(logger,tf.app.flags.FLAGS.flag_values_dict())

    try:
        trainer = Trainer(data_generator)
        if FLAGS.task == 'train':
            trainer.train()
            logger.info('save weights of model')
            if FLAGS.task_type in 'low':
                trainer.save_model_tfserving()
        elif FLAGS.task == 'predict':
            logger.info("do prediction, ext_is_predict_serving: d%", FLAGS.ext_is_predict_serving)
            trainer.predict()
        elif FLAGS.task == 'train_predict':
            trainer.train()
            logger.info('save weights of model')
            trainer.save_model_tfserving()
            logger.info("do prediction, ext_is_predict_serving: d%", FLAGS.ext_is_predict_serving)
            trainer.predict()

    except Exception as e:
        exc_info = traceback.format_exc(sys.exc_info())
        msg = 'creating session exception:%s\n%s' % (e, exc_info)
        tmp = 'Run called even after should_stop requested.'
        should_stop = type(e) == RuntimeError and str(e) == tmp
        if should_stop:
            logger.warn(msg)
        else:
            logger.error(msg)
        exit_code = 0 if should_stop else 1
        sys.exit(exit_code)



if __name__ == "__main__":
    logger.info("----start---")
    tf.app.run()

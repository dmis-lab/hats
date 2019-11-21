import os, time, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from logger import set_logger
from config import get_args
from dataset import StockDataset
from trainer import Trainer
from evaluator import Evaluator
from models.HATS import HATS

def init_prediction_model(config):
    with tf.variable_scope("model"):
        if config.model_type == "HATS":
            model = HATS(config)
    return model

def main():
    config = get_args()
    logger = set_logger(config)
    dataset = StockDataset(config)
    config.num_relations = dataset.num_relations
    config.num_companies = dataset.num_companies

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    model_name = config.model_type
    exp_name = '%s_%s_%s_%s_%s_%s_%s_%s'%(config.data_type, model_name,
                                        str(config.test_phase), str(config.test_size),
                                        str(config.train_proportion), str(config.lr),
                                        str(config.dropout), str(config.lookback))
    if not (os.path.exists(os.path.join(config.save_dir, exp_name))):
        os.makedirs(os.path.join(config.save_dir, exp_name))

    sess = tf.Session(config=run_config)
    model = init_prediction_model(config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    def model_summary(logger):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    model_summary(logger)
   
    #Training 
    evaluator = Evaluator(config, logger)
    trainer = Trainer(sess, model, dataset, config, logger, evaluator)
    trainer.train()
    
    #Testing
    loader = tf.train.Saver(max_to_keep=None)
    loader.restore(sess, tf.train.latest_checkpoint(os.path.join(config.save_dir, exp_name)))
    print("load best evaluation model")

    test_loss, report_all, report_topk = evaluator.evaluate(sess, model, dataset, 'test', trainer.best_f1['neighbors'])
    te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report_all
    logstr = 'EPOCH {} TEST ALL \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
            .format(trainer.best_f1['epoch'],test_loss,te_acc,te_cpt_acc,te_pred_rate,te_mac_f1,te_mic_f1,te_exp_rt)
    logger.info(logstr)

    te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report_topk
    logstr = 'EPOCH {} TEST TopK \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
            .format(trainer.best_f1['epoch'],test_loss,te_acc,te_cpt_acc,te_pred_rate,te_mac_f1,te_mic_f1,te_exp_rt)
    logger.info(logstr)

    #Print Log
    with open('%s_log.log'%model_name, 'a') as out_:
        out_.write("%d phase\n"%(config.test_phase))
        out_.write("%f\t%f\t%f\t%f\t%f\t%s\t%f\t%f\t%f\t%f\t%f\t%s\t%d\n"%(
            report_all[1], report_all[2], report_all[3], report_all[4], report_all[5], str(report_all[0]),
            report_topk[1], report_topk[2], report_topk[3], report_topk[4], report_topk[5], str(report_topk[0]),
            trainer.best_f1['epoch']))

if __name__ == '__main__':
    main()

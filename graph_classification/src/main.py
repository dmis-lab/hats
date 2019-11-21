import os, time, pdb, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from logger import set_logger
from config import get_args
from dataset import StockDataset

from graph_evaluator import GEvaluator
from graph_trainer import GTrainer
from evaluator import Evaluator
from trainer import Trainer

from models.HATS import HATS

def init_prediction_model(config):
    with tf.variable_scope('model'):
        if config.model_type == 'graph-HATS':
            model = HATS(config)
    return model

def main():
    config = get_args()
    logger = set_logger(config)
    dataset = StockDataset(config)
    config.num_relations = dataset.num_relations
    config.num_companies = dataset.num_companies
    run_config = tf.ConfigProto(log_device_placement=False)
    run_config.gpu_options.allow_growth = True
    exp_name = '%s_%s_%s_%s_%s_%s_%s_%s'%(config.data_type, config.model_type,
                                        str(config.test_phase), str(config.test_size),
                                        str(config.train_proportion), str(config.lr),
                                        str(config.dropout), str(config.lookback))
    # save train file
    if not (os.path.exists(os.path.join(config.save_dir, exp_name))):
        os.makedirs(os.path.join(config.save_dir, exp_name))

    sess = tf.Session(config=run_config)
    model = init_prediction_model(config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    def model_summary():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True) # print the name and shapes of the variables
    model_summary()

    if config.mode == 'train':
        if 'graph' in config.model_type:
            evaluator = GEvaluator(config, logger)
            trainer = GTrainer(sess, model, dataset, config, logger, evaluator)
        else:
            evaluator = Evaluator(config, logger)
            trainer = Trainer(sess, model, dataset, config, logger, evaluator)
    trainer.train()

    #Testing
    loader = tf.train.Saver(max_to_keep=None)
    loader.restore(sess, tf.train.latest_checkpoint(os.path.join(config.save_dir, exp_name)))
    print("load best evaluation model")
    test_loss, report = evaluator.evaluate(sess, model, dataset, 'test')
    te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report
    logstr = 'EPOCH {} TEST ALL \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
            .format(trainer.best_f1['epoch'],test_loss,te_acc,te_cpt_acc,te_pred_rate,te_mac_f1,te_mic_f1,te_exp_rt)
    logger.info(logstr)

    with open('%s_log.log'%config.GNN_model+'_'+config.model_type+'_'+config.data_type+'_'+config.price_model+'_'+str(config.test_phase), 'a') as out_:
        out_.write("%d phase\n"%(config.test_phase))
        out_.write("%f\t%f\t%f\t%f\t%f\t%s\t%d\n"%(
            report[1], report[2], report[3], report[4], report[5], str(report[0]),
            trainer.best_f1['epoch']))

if __name__ == "__main__":
    main()

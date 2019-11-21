from base.base_train import BaseTrain
import numpy as np
import pdb, time
import tensorflow as tf

class GTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger, evaluator):
        super(GTrainer, self).__init__(sess, model, data, config, logger, evaluator)
        self.keep_prob = 1-config.dropout

    def train_epoch(self, ):
        # error NoneType object is not an iterator
        all_x, all_y, all_rt = next(self.data.get_batch('train', self.config.lookback))
        losses, labels, preds, probs, rts = list(), list(), list(), list(), list()
        accs, cpt_accs, pred_rates, mac_f1, mic_f1, exp_rts = [], [], [], [], [], []
        for b_idx, (x, y, rt) in enumerate(zip(all_x, all_y, all_rt)):
            loss, label, pred, prob, rt = self.train_step(x, y, rt, b_idx)

            losses.append(loss)
            labels.append(label)
            preds.append(pred)
            probs.append(prob)
            rts.append(rt)

        report = self.evaluator.metric(np.concatenate(labels), np.concatenate(preds), np.concatenate(probs), np.concatenate(rts))
        loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)

        return loss, report

    def create_feed_dict(self, x, y):
        if self.config.GNN_model=='GCN':
            feed_dict = {self.model.x: x, self.model.y: y,
                        self.model.rel_mat:self.data.rel_mat,
                        self.model.keep_prob: self.keep_prob}
        elif self.config.GNN_model=='HATS':
            feed_dict = {self.model.x: x, self.model.y: y,
                        self.model.rel_mat:self.data.neighbors, # [Nul rels, Max k]
                        self.model.rel_num:self.data.rel_num, # [Nul rels, Max k]
                        self.model.keep_prob: self.keep_prob}
        elif self.config.GNN_model=='G-HATS':
            feed_dict = {self.model.x: x, self.model.y: y,
                        self.model.rel_mat:self.data.rel_nodes, # [Nul rels, Max k]
                        self.model.rel_num:self.data.rel_num, # [Nul rels, Max k]
                        self.model.keep_prob: self.keep_prob}
        elif self.config.GNN_model == 'TGC':
            rel_multi_hot = self.data.rel_multi_hot
            rel_mask = (np.diag([-1]*rel_multi_hot.shape[0]) + 1) * (-1e-9)

            feed_dict = {self.model.x: x, self.model.y: y,
                        self.model.relation:rel_multi_hot,
                        self.model.rel_mask:rel_mask,
                        self.model.keep_prob: self.keep_prob}
        return feed_dict

    def train_step(self, x, y, rt, b_idx):
        # batch is whole dataset of a single company
        feed_dict = self.create_feed_dict(x,y)

        _, loss, pred, prob = self.sess.run([self.model.train_step, self.model.cross_entropy,
                                    self.model.prediction, self.model.prob],
                                     feed_dict=feed_dict)

        label = np.argmax(y, 1)

        return loss, label, pred, prob, rt

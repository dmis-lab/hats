from base.base_train import BaseTrain
import numpy as np
import time, random
import tensorflow as tf

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger, evaluator):
        super(Trainer, self).__init__(sess, model, data, config, logger, evaluator)
        self.keep_prob = 1-config.dropout

    def sample_neighbors(self):
        k = self.config.neighbors_sample
        if self.config.model_type == 'HATS':
            neighbors_batch = []
            for rel_neighbors in self.data.neighbors:
                rel_neighbors_batch = []
                for cpn_idx, neighbors in enumerate(rel_neighbors):
                    short = max(0, k-neighbors.shape[0])
                    if short: # less neighbors than k
                        neighbors = np.expand_dims(np.concatenate([neighbors, np.zeros(short)]),0)
                        rel_neighbors_batch.append(neighbors)
                    else:
                        neighbors = np.expand_dims(np.random.choice(neighbors, k),0)
                        rel_neighbors_batch.append(neighbors)
                neighbors_batch.append(np.expand_dims(np.concatenate(rel_neighbors_batch,0),0))

        return np.concatenate(neighbors_batch,0)

    def train_epoch(self):
        all_x, all_y, all_rt = next(self.data.get_batch('train', self.config.lookback))
        neighbors = self.sample_neighbors()
        labels = []
        losses, accs, cpt_accs, pred_rates, mac_f1, mic_f1, exp_rts = [], [], [], [], [], [], []
        accs_k, cpt_accs_k, pred_rates_k, mac_f1_k, mic_f1_k, exp_rts_k = [], [], [], [], [], []

        for x, y, rt in zip(all_x, all_y, all_rt):
            loss, metrics = self.train_step(x, y, rt, neighbors)

            metrics_all, metrics_topk = metrics
            losses.append(loss)
            pred_rates.append(metrics_all[0])
            accs.append(metrics_all[1][0])
            cpt_accs.append(metrics_all[1][1])
            mac_f1.append(metrics_all[2])
            mic_f1.append(metrics_all[3])
            exp_rts.append(metrics_all[4])
            pred_rates_k.append(metrics_topk[0])
            accs_k.append(metrics_topk[1][0])
            cpt_accs_k.append(metrics_topk[1][1])
            mac_f1_k.append(metrics_topk[2])
            mic_f1_k.append(metrics_topk[3])
            exp_rts_k.append(metrics_topk[4])

        report_all = [np.around(np.array(pred_rates).mean(0),decimals=4), np.mean(accs), np.mean(cpt_accs), np.mean(mac_f1), np.mean(mic_f1), np.mean(exp_rts)]
        report_topk = [np.around(np.array(pred_rates_k).mean(0),decimals=4), np.mean(accs_k), np.mean(cpt_accs_k), np.mean(mac_f1_k), np.mean(mic_f1_k), np.mean(exp_rts_k)]
        loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        return loss, report_all, report_topk


    def get_rel_multi_hot(self, batch_neighbors):
        neighbors_multi_hot = []
        for cpn_idx, neighbors in enumerate(batch_neighbors):
            multi_hots = []
            for n_i in neighbors:
                if n_i ==0:
                    multi_hots.append(np.expand_dims(self.data.rel_multi_hot[cpn_idx, cpn_idx],0)) #all zeros
                else:
                    multi_hots.append(np.expand_dims(self.data.rel_multi_hot[cpn_idx, int(n_i)],0))
            neighbors_multi_hot.append(np.expand_dims(np.concatenate(multi_hots,0),0))
        return np.concatenate(neighbors_multi_hot,0)


    def create_feed_dict(self, x, y, neighbors):
        if self.config.model_type == 'HATS':
            feed_dict = {self.model.x: x, self.model.y: y,
                        self.model.rel_num:self.data.rel_num,
                        self.model.rel_mat:neighbors,
                        self.model.max_k:neighbors.shape[-1],
                        self.model.keep_prob: self.keep_prob}

        return feed_dict

    def train_step(self, x, y, rt, neighbors):
        # batch is whole dataset of a single company
        feed_dict = self.create_feed_dict(x, y, neighbors)
        _, loss, pred, prob = self.sess.run([self.model.train_step, self.model.cross_entropy,
                                    self.model.prediction, self.model.prob],
                                     feed_dict=feed_dict)
        label = np.argmax(y, 1)
        metrics = self.evaluator.metric(label, pred, prob, rt)

        return loss, metrics

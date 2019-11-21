
import pdb, math
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class GEvaluator:
    def __init__(self, config, logger):
        self.config = config
        self.keep_prob = 1-config.dropout
        self.logger = logger
        self.n_labels = len(config.label_proportion)

    def create_feed_dict(self, model, data, x, y):
        if self.config.GNN_model=='GCN':
            feed_dict = {model.x: x, model.y: y,
                        model.rel_mat: data.rel_mat,
                        model.keep_prob: self.keep_prob}
        elif self.config.GNN_model=='HATS':
            feed_dict = {model.x: x, model.y: y,
                        model.rel_mat: data.neighbors, # [Nul rels, Max k]
                        model.rel_num: data.rel_num, # [Nul rels, Max k]
                        model.keep_prob: self.keep_prob}
        elif self.config.GNN_model=='G-HATS':
            feed_dict = {model.x: x, model.y: y,
                        model.rel_mat:data.rel_nodes, # [Nul rels, Max k]
                        model.rel_num:data.rel_num, # [Nul rels, Max k]
                        model.keep_prob: self.keep_prob}
        elif self.config.GNN_model == 'TGC':
            rel_multi_hot = data.rel_multi_hot
            rel_mask = (np.diag([-1]*rel_multi_hot.shape[0]) + 1) * (-1e-9)

            feed_dict = {model.x: x, model.y: y,
                        model.relation:rel_multi_hot,
                        model.rel_mask:rel_mask,
                        model.keep_prob: self.keep_prob}
        return feed_dict

    def evaluate(self, sess, model, data, phase):
        all_x, all_y, all_rt = next(data.get_batch(phase, self.config.lookback))
        # in evaluation batch is whole dataset of a single company
        losses, labels, preds, probs, rts = list(), list(), list(), list(), list()
        for b_idx, (x, y, rt) in enumerate(zip(all_x, all_y, all_rt)):
            feed_dict = self.create_feed_dict(model, data,x,y)
            # feed_dict[model.state] = self.state_dict[phase][b_idx]
            loss, pred, prob= sess.run([model.cross_entropy, model.prediction, model.prob],
                                        feed_dict=feed_dict)

            label = np.argmax(y, 1)
            losses.append(loss)
            labels.append(label)
            preds.append(pred)
            probs.append(prob)
            rts.append(rt)

        report = self.metric(np.concatenate(labels), np.concatenate(preds), np.concatenate(probs), np.concatenate(rts))

        return np.mean(losses), report

    def get_f1(self, y, y_):
        # y : label | y_ : pred
        return f1_score(y,y_,average='macro'), f1_score(y,y_,average='micro')

    def get_acc(self, conf_mat):
        accuracy = conf_mat.trace()/conf_mat.sum()
        if self.n_labels==2:
            compact_accuracy = accuracy
        else:
            # It is actually a recall of up down
            compact_conf_mat = np.take(conf_mat,[[0,2],[6,8]])
            compact_accuracy = compact_conf_mat.trace()/compact_conf_mat.sum()
        return accuracy, compact_accuracy

    def expected_return(self, pred, prob, returns):
        # To create neuralized portfolio
        n_mid = prob.shape[0]//2
        # sorted : ascending order (based on down probabilty)
        # both side have exactly the half size of the universe
        short_half_idx = np.argsort(prob[:,0])[-n_mid:]
        long_half_idx = np.argsort(prob[:,-1])[-n_mid:]
        # if prediction was neutral, we don'y count it as our return
        short_rts = (returns[short_half_idx]*(pred[short_half_idx]==0)).mean() * (-1)
        long_rts = (returns[long_half_idx]*(pred[long_half_idx]==(self.n_labels-1))).mean()
        return (short_rts + long_rts)

    def filter_topk(self, label, pred, prob, returns, topk):
        short_k_idx = np.argsort(prob[:,0])[-topk:]
        long_k_idx = np.argsort(prob[:,-1])[-topk:]
        topk_idx = np.concatenate([short_k_idx, long_k_idx])
        return label[topk_idx], pred[topk_idx], prob[topk_idx], returns[topk_idx]

    def cal_metric(self, label, pred, prob, returns):
        exp_returns = self.expected_return(pred, prob, returns)
        conf_mat = confusion_matrix(label, pred, labels=[i for i in range(self.n_labels)])
        acc, cpt_acc = self.get_acc(conf_mat)
        mac_f1, mic_f1 = self.get_f1(label, pred)
        pred_rate = [(pred==i).sum()/pred.shape[0] for i in range(self.n_labels)]
        return pred_rate, acc, cpt_acc, mac_f1, mic_f1, exp_returns

    def metric(self, label, pred, prob, returns, topk=10):
        metric_all = self.cal_metric(label, pred, prob, returns)
        return metric_all

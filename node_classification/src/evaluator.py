import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        self.n_labels = len(config.label_proportion)

    def sample_neighbors(self, data):
        k = self.config.neighbors_sample
        if self.config.model_type == 'HATS':
            neighbors_batch = []
            for rel_neighbors in data.neighbors:
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

    def get_rel_multi_hot(self, batch_neighbors, data):
        neighbors_multi_hot = []
        for cpn_idx, neighbors in enumerate(batch_neighbors):
            multi_hots = []
            for n_i in neighbors:
                if n_i ==0:
                    multi_hots.append(np.expand_dims(data.rel_multi_hot[cpn_idx, cpn_idx],0)) #all zeros
                else:
                    multi_hots.append(np.expand_dims(data.rel_multi_hot[cpn_idx, int(n_i)],0))
            neighbors_multi_hot.append(np.expand_dims(np.concatenate(multi_hots,0),0))
        return np.concatenate(neighbors_multi_hot,0)

    def create_feed_dict(self, model, data, x, y, phase, neighbors=None):
        if self.config.model_type == 'HATS':
            feed_dict = {model.x: x, model.y: y,
                        model.rel_num:data.rel_num,
                        model.rel_mat:neighbors,
                        model.max_k:neighbors.shape[-1]}
        return feed_dict
    
    def get_result(self, sess, model, data, phase, neighbors=None):
        all_x, all_y, all_rt = next(data.get_batch(phase, self.config.lookback))
        preds = list()
        probs = list()
        for b_idx, (x, y, rt) in enumerate(zip(all_x, all_y, all_rt)):
            feed_dict = self.create_feed_dict(model, data, x, y, phase, neighbors)
            pred, prob = sess.run([model.prediction, model.prob], feed_dict=feed_dict)
            preds.append(pred)
            probs.append(prob)
        return np.array(preds), np.array(probs)

    def evaluate(self, sess, model, data, phase, neighbors=None):
        all_x, all_y, all_rt = next(data.get_batch(phase, self.config.lookback))
        # in evaluation batch is whole dataset of a single company
        losses, accs, cpt_accs, pred_rates, mac_f1, mic_f1, exp_rts = [], [], [], [], [], [], []
        accs_k, cpt_accs_k, pred_rates_k, mac_f1_k, mic_f1_k, exp_rts_k = [], [], [], [], [], []
        for x, y, rt in zip(all_x, all_y, all_rt):
            feed_dict = self.create_feed_dict(model, data, x, y, phase, neighbors)

            loss, pred, prob = sess.run([model.cross_entropy, model.prediction, model.prob],
                                        feed_dict=feed_dict)
            label = np.argmax(y, 1)
            metrics_all, metrics_topk = self.metric(label, pred, prob, rt)

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
        return np.mean(losses), report_all, report_topk


    def create_confusion_matrix(self, y, y_, is_distribution=False):
        n_samples = float(y_.shape[0])   # get dimension list
        if is_distribution:
            label_ref = np.argmax(y_, 1)  # 1-d array of 0 and 1
            label_hyp = np.argmax(y, 1)
        else:
            label_ref, label_hyp = y, y_

        # p & n in prediction
        p_in_hyp = np.sum(label_hyp)
        n_in_hyp = n_samples - p_in_hyp

        # Positive class: up
        tp = np.sum(np.multiply(label_ref, label_hyp))  # element-wise, both 1 can remain
        fp = p_in_hyp - tp  # predicted positive, but false

        # Negative class: down
        tn = n_samples - np.count_nonzero(label_ref + label_hyp)  # both 0 can remain
        fn = n_in_hyp - tn  # predicted negative, but false
        return float(tp), float(fp), float(tn), float(fn)

    def get_mcc(self, tp, fp, tn, fn):
        core_de = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return (tp * tn - fp * fn) / math.sqrt(core_de) if core_de else 0

    def get_f1(self, tp, fp, tn, fn):
        eps = 1e-10
        precision = tp / (tp+fp+eps)
        recall = tp / (tp+fn+eps)
        return 2 * (precision*recall) / (precision+recall+eps)

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
        return (short_rts + long_rts) * 100

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
        return pred_rate, (acc, cpt_acc), mac_f1, mic_f1, exp_returns

    def metric(self, label, pred, prob, returns, topk=30):
        metric_all = self.cal_metric(label, pred, prob, returns)
        label, pred, prob, returns = self.filter_topk(label, pred, prob, returns, topk)
        metric_topk = self.cal_metric(label, pred, prob, returns)
        return metric_all, metric_topk

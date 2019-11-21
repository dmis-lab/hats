import os, random, pdb, math
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataset():
    def __init__(self, config):
        self.config = config
        self.model_type = config.model_type
        self.mkt_data_dir = os.path.join(config.market_data_dir, config.data_type)
        self.rel_data_dir = config.relation_data_dir
        self.use_rel_list = config.use_rel_list
        self.data_type = config.data_type
        self.min_train_period = config.min_train_period
        self.lookback = config.lookback
        self.neighbors_sample = config.neighbors_sample
        self.feature_list = config.feature_list
        self.scaling_feats = [idx for idx, name in enumerate(self.feature_list) if name!='return']
        self.n_labels = len(config.label_proportion)
        self.train_set, self.test_set, self.dev_set = [], [], []
        self.train_label, self.test_label, self.dev_label = [], [], []

        self.scaler = MinMaxScaler()
        self.read()

    def read(self, ):
        # read relation data
        with open(os.path.join(self.rel_data_dir, 'ordered_ticker.pkl'), 'rb') as f:
            ordered_ticker = pickle.load(f)
        if self.config.model_type == 'HATS':
            with open(os.path.join(self.rel_data_dir, 'adj_mat.pkl'), 'rb') as f:
                self.rel_mat = pickle.load(f)[self.use_rel_list]
                #self.rel_mat = pickle.load(f)[self.use_rel_list][:,:,:self.neighbors_sample]
                self.neighbors = []
                for rel_idx, rel_mat_i in enumerate(self.rel_mat):
                    rel_neighbors = []
                    for cpn_idx, row in enumerate(rel_mat_i):
                        rel_neighbors.append(row.nonzero()[0])
                    self.neighbors.append(rel_neighbors)

        with open(os.path.join(self.rel_data_dir, 'rel_num.pkl'), 'rb') as f:
            self.rel_num = pickle.load(f)[self.use_rel_list]
            self.num_relations = len(self.rel_num)
        if self.config.model_type == 'HATS':
            k = self.neighbors_sample
            self.rel_num = (self.rel_num>=k)*k + (self.rel_num<k)*self.rel_num

        tot_ph = int(1652/self.config.test_size)
        if tot_ph < self.config.train_proportion + 1:
            print("Unsuitable train-test size")
            raise
        model_phase = math.ceil(self.config.test_phase + self.config.train_proportion)
        if model_phase >= tot_ph:
            print("Unsuitable test phase model_phase:%d | total_pahse:%d"%(model_phase, tot_ph))
            raise
        train_size = int(self.config.test_size * self.config.train_proportion) - self.config.dev_size
        test_start_idx = model_phase * self.config.test_size

        # read market data
        all_rt = []
        i = 0
        for ticker in ordered_ticker:
            df = pd.read_csv(os.path.join(self.mkt_data_dir,'processed', '{}_data.csv'.format(ticker)))
            if df.shape[0] != 1652:
                print(ticker, df.shape)
                i+=1
            test_target_start_idx = test_start_idx
            test_input_start_idx = test_target_start_idx - self.lookback
            dev_target_start_idx = test_start_idx - self.config.dev_size
            dev_input_start_idx = test_target_start_idx - self.lookback - self.config.dev_size

            label_df = df['return']
            df = df[self.feature_list]

            self.train_set.append(df.iloc[:dev_target_start_idx].values[-train_size-self.lookback:])
            self.train_label.append(np.expand_dims(label_df.iloc[:dev_target_start_idx].values[-train_size-self.lookback:], 1))

            self.dev_set.append(df.iloc[:test_target_start_idx].values[-self.config.dev_size-self.lookback:])
            self.dev_label.append(np.expand_dims(label_df.iloc[:test_target_start_idx].values[-self.config.dev_size-self.lookback:], 1))

            self.test_set.append(df.iloc[test_input_start_idx:test_target_start_idx+self.config.test_size].values)
            self.test_label.append(np.expand_dims(label_df.iloc[test_input_start_idx:test_target_start_idx+self.config.test_size].values,1))

        self.num_companies = len(self.train_label)
        # Params for normalize
        all_tr_rt = []
        for tr_set in self.train_label:
            all_tr_rt+=tr_set.tolist()
        self.tr_mean = np.mean(all_tr_rt)
        self.tr_std = np.std(all_tr_rt)
        self.threshold = list()
        th_tot = np.sum(self.config.label_proportion)
        tmp_rt = np.sort(all_tr_rt, axis=0)
        tmp_th = 0
        for th in self.config.label_proportion:
            self.threshold.append(tmp_rt[int(len(all_tr_rt)*float(th+tmp_th)/th_tot-1)][0])
            tmp_th += th

        # check statistics
        tr_rt, dev_rt, te_rt = [], [], []
        for tr_set in self.train_label:
            tr_rt+=tr_set.tolist()
        for dev_set in self.dev_label:
            dev_rt+=dev_set.tolist()
        for te_set in self.test_label:
            te_rt+=te_set.tolist()

        tr_rate = [(np.array(tr_rt)<th).sum()/len(tr_rt) for th in self.threshold[:-1]]
        dev_rate = [(np.array(dev_rt)<th).sum()/len(dev_rt) for th in self.threshold[:-1]]
        te_rate = [(np.array(te_rt)<th).sum()/len(te_rt) for th in self.threshold[:-1]]

        print('Train label rate {}'.format(tr_rate))
        print('development label rate {}'.format(dev_rate))
        print('Test label rate {}'.format(te_rate))

    def get_batch(self, phase, lookback):
        if phase == 'train':
            # Randomly create train batch when train
            if self.config.train_on_stock:
                # batch shape [Num Companies, Sequence length, Feature Dim]
                batch, label = [], []
                for data, label_data in zip(self.train_set, self.train_label):
                    one_batch, one_label = self.batch_for_one_stock(data, label_data, lookback)
                    one_batch = self.formatting(one_batch)
                    if len(one_batch) !=0:
                        batch.append(one_batch)
                        label.append(one_label)
            else:
                # batch shape [Sequence length, Num Companies, Feature Dim]
                batch, label, returns = self.batch_for_aligned_days(self.train_set, self.train_label, lookback)

        elif phase == 'eval':
            if self.config.train_on_stock:
                batch, label = [], []
                for data, label_data in zip(self.dev_set, self.dev_label):
                    one_batch, one_label, one_return = self.batch_for_one_stock(data, label_data, lookback)
                    one_batch = self.formatting(one_batch)
                    if len(one_batch) !=0:
                        batch.append(one_batch)
                        label.append(one_label)
            else:
                batch, label, returns = self.batch_for_aligned_days(self.dev_set, self.dev_label, lookback)

        elif phase == 'test':
            if self.config.train_on_stock:
                batch, label = [], []
                for data, label_data in zip(self.test_set, self.test_label):
                    one_batch, one_label, one_return = self.batch_for_one_stock(data, label_data, lookback)
                    one_batch = self.formatting(one_batch)
                    if len(one_batch) !=0:
                        batch.append(one_batch)
                        label.append(one_label)
            else:
                batch, label, returns = self.batch_for_aligned_days(self.test_set, self.test_label, lookback)

        else:
            raise("Not proper phase %s"%phase)

        yield batch, label, returns

    def batch_for_one_stock(self, data, label_data, lookback):
        one_batch, one_label = [], []
        all_mv = []
        for w_start in range(data.shape[0]-lookback-1):
            mv_percent = label_data[w_start+lookback]
            all_mv.append(mv_percent)
            mv_class = self.classify(mv_percent)
            if not mv_class: # not valid data sample
                continue
            # Use feature untiil one day before
            window = data[w_start:w_start+lookback]
            if len(self.scaling_feats):
                non_cols = list(set([i for i in range(len(self.feature_list))]) - set(self.scaling_feats))
                non = self.tr_normalize(window[:, non_cols])
                scale = self.normalize(window[:,self.scaling_feats].reshape(-1, len(self.scaling_feats)))
                window = np.concatenate([non, scale], 1)
            else:
                window = self.tr_normalize(window)
            one_batch.append(window)
            one_return.append(mv_percent)
            one_label.append(mv_class)

        return np.array(one_batch), np.array(one_label), np.array(one_return)

    def batch_for_aligned_days(self, data, label_data, lookback):
        batch, label, returns = [], [], []
        all_mv = []
        # [Companies, Sequence length, Feature Dims]
        data = np.array(data)
        label_data = np.concatenate(label_data, 1)

        for w_start in range(data.shape[1]-lookback-1):
            mv_percent = label_data[w_start+lookback]
            all_mv.append(mv_percent)

            mv_class = np.zeros([len(mv_percent), self.n_labels])

            if self.n_labels == 2:
                mv_class[mv_percent<self.threshold[0]] = [1.,0.]
                mv_class[mv_percent>=self.threshold[0]] = [0.,1.]
            elif self.n_labels == 3:
                mv_class[mv_percent<self.threshold[0]] = [1.,0.,0.]
                mv_class[(mv_percent>=self.threshold[0]) & (mv_percent<self.threshold[1])] = [0.,1.,0.]
                mv_class[mv_percent>=self.threshold[1]] = [0.,0.,1.]

            # Use feature until one day before
            window = data[:,w_start:w_start+lookback]
            if len(self.scaling_feats):
                non_cols = list(set([i for i in range(len(self.feature_list))]) - set(self.scaling_feats))
                non = self.tr_normalize(window[:,:,non_cols])
                scale = self.normalize(window[:,:,self.scaling_feats])
                window = np.concatenate([non, scale], 2)
            else:
                window = self.tr_normalize(window)
            batch.append(self.formatting(window))
            label.append(mv_class)
            returns.append(mv_percent)
        return batch, label, returns

    def minmax_scaling(self, batch):
        max_ = batch.max(0)
        min_ = batch.min(0)
        return (batch - min_) / (max_-min_)

    def normalize(self, batch):
        return (batch-np.expand_dims(batch.mean(1),1))/np.expand_dims(batch.std(1),1)

    def tr_normalize(self, batch):
        return (batch-self.tr_mean)/self.tr_std

    def classify(self, mv_percent):
        if mv_percent < self.threshold[0]: # valid down sample
            mv_class = [1, 0]
        elif mv_percent >= self.threshold[-2]: # valid up sample
            mv_class = [0, 1]
        else:
            mv_class = [0, 0]
        return mv_class

    def formatting(self, batch):
        batch_size, lookback = batch.shape[0], batch.shape[1]
        batch = np.reshape(batch, [batch_size, lookback, len(self.feature_list)])
        return batch

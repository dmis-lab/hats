import os, random, pdb, math
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import get_args
from logger import set_logger

class StockDataset():
    def __init__(self, config):
        self.config = config
        self.model_type = config.model_type
        self.mkt_data_dir = config.market_data_dir
        self.rel_data_dir = config.relation_data_dir
        self.data_type = config.data_type
        self.min_train_period = config.min_train_period
        self.lookback = config.lookback
        self.feature_list = config.feature_list
        self.scaling_feats = [idx for idx, name in enumerate(self.feature_list) if name != 'return']
        self.n_labels = len(config.label_proportion)
        self.dev_size = config.dev_size
        self.test_size = config.test_size
        self.train_set, self.test_set, self.dev_set = list(), list(), list()
        self.train_label, self.test_label, self.dev_label = list(), list(), list()
        self.num_relations = 0
        self.num_companies = 0

        self.scaler = MinMaxScaler()
        if config.preprocess:
            self.preprocess()
        self.read()

    def preprocess(self,):
        raw_files = os.listdir(os.path.join(self.mkt_data_dir,'raw'))
        for file in raw_files :
            df = pd.read_csv(os.path.join(self.mkt_data_dir,'raw',file))
            if self.data_type == 'snp500':
                df.columns = ['date','open','high','low','close','volume', 'name']
                df = df.fillna(method='ffill')

            elif self.data_type == 'stocknet': # stocknet 88 companies
                df.columns = ['date','open','high','low','close','adj_close','volume']
                df = df.fillna(method='ffill')

            elif self.data_type == 'graph_pooling':
                df.columns = ['date','open','high','low','close','volume']
                df = df.fillna(method='ffill')

            df = df[['date','open','high','low','close','volume']]
            close = df['close'].values

            # calculate return
            pt_1 = close[:-1]
            pt = close[1:]
            rt  = (pt / pt_1) -1
            df['return'] = 0
            df['return'].iloc[1:] = rt

            df.to_csv(os.path.join(self.mkt_data_dir,'processed',file), index=False)
            print(file,'done')

    def read(self, ):
        # read relation data and stock data

        with open(self.rel_data_dir + self.data_type + '_ordered_ticker.pkl', 'rb') as fp:
            ordered_ticker = pickle.load(fp)
        with open(self.rel_data_dir + self.data_type + '_adj_mat.pkl', 'rb') as fp:
            self.rel_mat = pickle.load(fp)
        self.num_companies = len(ordered_ticker)-1

        if self.config.GNN_model=='HATS':
            self.rel_mat = self.rel_mat[:,:,:20]
            self.rel_num, self.neighbors = [], []
            for rel_idx, rel_mat_i in enumerate(self.rel_mat):
                r_num = np.zeros([1, self.num_companies])
                r_nb = np.zeros([self.num_companies, 20]) # sample 20 only
                rel_neighbors = []
                for cpn_idx, row in enumerate(rel_mat_i):
                    val_nb = row.nonzero()[0].tolist()
                    if len(val_nb) != 0:
                        r_nb[cpn_idx,:len(val_nb)] = val_nb
                    r_num[0,cpn_idx] = len(val_nb)
                self.neighbors.append(np.expand_dims(r_nb,0))
                self.rel_num.append(r_num)
            self.rel_num = np.concatenate(self.rel_num, 0)
            self.neighbors = np.concatenate(self.neighbors, 0)
            self.num_relations = len(self.rel_num)

        label_df = pd.read_csv(os.path.join(self.mkt_data_dir, '{}.csv'.format(self.data_type)))
        y_data_date = [label_df.date.iloc[0],label_df.date.iloc[-1]]
        date_len = label_df.loc[(label_df.date>=y_data_date[0]) & (label_df.date<=y_data_date[1])].shape[0]
        tot_ph = int(date_len / self.config.test_size)
        if tot_ph < self.config.train_proportion + 1:
            print ("Unsuitable train-test size")
            raise
        model_phase = math.ceil(self.config.test_phase + self.config.train_proportion)
        if model_phase >= tot_ph:
            print("Unsuitable test phase model_phase:%d | total_pahse:%d"%(model_phase, tot_ph))
            raise
        train_size = int(self.config.test_size * self.config.train_proportion)
        test_start_idx = model_phase * self.config.test_size
        test_target_start_idx = test_start_idx
        test_input_start_idx = test_target_start_idx - self.lookback
        dev_target_start_idx = test_start_idx - self.config.dev_size
        dev_input_start_idx = test_target_start_idx - self.lookback - self.config.dev_size

        all_rt = []
        self.train_label.append(np.expand_dims(label_df.iloc[:dev_target_start_idx]['return'].values[-train_size-self.lookback:], 1))
        self.dev_label.append(np.expand_dims(label_df.iloc[:test_target_start_idx]['return'].values[-self.dev_size-self.lookback:], 1))
        self.test_label.append(np.expand_dims(label_df.iloc[test_input_start_idx:]['return'].values[-self.test_size-self.lookback:], 1))

        for ticker in ordered_ticker:
            if ticker == self.config.data_type:
                continue
            df = pd.read_csv(os.path.join(self.mkt_data_dir, 'snp500/processed', '{}_data.csv'.format(ticker)))
            df = df.loc[(df.date>=y_data_date[0]) & (df.date<=y_data_date[1])]
            df = df[self.feature_list]

            self.train_set.append(df.iloc[:dev_target_start_idx].values[-train_size-self.lookback:])
            self.dev_set.append(df.iloc[:test_target_start_idx].values[-self.dev_size-self.lookback:])
            self.test_set.append(df.iloc[test_input_start_idx:test_target_start_idx+self.test_size].values)

        all_tr_rt = list()
        for tr_set in self.train_label:
            all_tr_rt += tr_set.tolist()
        self.tr_mean =  np.mean(all_tr_rt)
        self.tr_std = np.std(all_tr_rt)
        self.threshold = list()
        th_tot = np.sum(self.config.label_proportion)
        tmp_rt = np.sort(all_tr_rt, axis=0)
        tmp_th = 0

        for th in self.config.label_proportion:
            self.threshold.append(tmp_rt[int(len(all_tr_rt) * float(th+tmp_th) / th_tot - 1)][0])
            tmp_th += th

        # check statistics
        tr_rt, ev_rt, te_rt = [], [], []
        for tr_set in self.train_label:
            tr_rt+=tr_set.tolist()
        for ev_set in self.dev_label:
            ev_rt+=ev_set.tolist()
        for te_set in self.test_label:
            te_rt+=te_set.tolist()

        tr_rate = [(np.array(tr_rt)<th).sum()/len(tr_rt) for th in self.threshold[:-1]]
        ev_rate = [(np.array(ev_rt)<th).sum()/len(ev_rt) for th in self.threshold[:-1]]
        te_rate = [(np.array(te_rt)<th).sum()/len(te_rt) for th in self.threshold[:-1]]

        print('Train label rate {}'.format(tr_rate))
        print('Dev label rate {}'.format(ev_rate))
        print('Test label rate {}'.format(te_rate))

    def get_batch(self, phase, lookback):
        if phase == 'train':
            # randomly create train batch when train
            if self.config.train_on_stock:
                # [Num companies, Sequence Length, Feature Dim]
                batch, label = list(), list()
                for data, label_data in zip(self.train_set, self.train_label):
                    one_batch, one_label = self.batch_for_one_stock(data, label_data, lookback)
                    one_batch = self.formatting(one_batch)
                    if len(one_batch) != 0:
                        batch.append(one_batch)
                        label.append(one_label)
            else:
                # [Sequence Length, Num Companies, Feature Dim]
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
        pass

    def batch_for_aligned_days(self, data, label_data, lookback):
        batch, label, returns = list(), list(), list()
        # [Num Companies, Sequence Length, Feature Dims]
        label_data = np.array(label_data)
        num_comp = 0
        data = np.array(data)
        data = np.concatenate([label_data, data], 0)
        data = label_data

        for w_start in range(data.shape[1]-lookback-1):
            mv_percent = label_data[0][w_start+lookback]
            mv_class = np.zeros([len(mv_percent), self.n_labels])

            if self.n_labels == 2:
                mv_class[mv_percent < self.threshold[0]] = [1., 0.]
                mv_class[mv_percent >= self.threshold[0]] = [0., 1.]
            elif self.n_labels == 3:
                mv_class[mv_percent<self.threshold[0]] = [1.,0.,0.]
                mv_class[(mv_percent>=self.threshold[0]) & (mv_percent<self.threshold[1])] = [0.,1.,0.]
                mv_class[mv_percent>=self.threshold[1]] = [0.,0.,1.]

            window = data[:, w_start:w_start+lookback]
            if self.scaling_feats:
                non_cols = list(set([i for i in range(len(self.feature_list))]) - set(self.scaling_feats))
                non = self.tr_normalize(window[:, :, non_cols])
                scale = self.normalize(window[:, :, self.scaling_feats])
                window = np.concatenate([non, scale], 2)
            else:
                window = self.tr_normalize(window) # (22, 20, 1)

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
        if ('lstm' in self.config.model_type) or ('graph' in self.config.model_type):
            batch = np.reshape(batch, [batch_size, lookback, len(self.feature_list)])
        else:
            batch = np.reshape(batch, [batch_size, lookback*len(self.feature_list)])
        return batch

# for sanity check
def main():
    config = get_args()
    logger = set_logger(config)
    dataset = StockDataset(config)

if __name__ == "__main__":
    main()

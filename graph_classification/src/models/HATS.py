import tensorflow as tf
import numpy as np
import os
import random
import re
import time
import pdb

from config import get_args
from base.base_model import BaseModel

class HATS(BaseModel):
    def __init__(self, config):
        super(HATS, self).__init__(config)
        self.GNN_model = config.GNN_model
        self.input_dim = len(config.feature_list)
        self.keep_prob = 1-config.dropout
        self.node_features = config.node_features
        self.use_bias = config.use_bias
        self.stack_layer = config.stack_layer
        self.num_layer = config.num_layer
        self.n_labels = len(config.label_proportion)
        self.num_relations = config.num_relations
        self.num_companies = config.num_companies

        self.build_model()
        self.init_saver()

    def get_state(self, price_model=None):
        # 2d conv input shape [ch_in, batch_size, H, W]
        with tf.variable_scope('feat_ext_ops'):
            # price model = LSTM
            if 'LSTM' in price_model:
                cells = [tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True) for _ in range(self.num_layer)]
                dropout = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob,
                                            output_keep_prob=self.keep_prob) for cell in cells]
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(dropout, state_is_tuple=True)
                outputs, state = tf.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.float32) # (?, lookback, 64), (?, 64)
                return state[-1][-1]

    def create_relation_onehot(self, ):
        one_hots = []
        for rel_idx in range(self.num_relations):
            one_hots.append(tf.one_hot([rel_idx],depth=self.num_relations))
        return tf.concat(one_hots,0)

    def matmul(self,x, y, sparse=False):
        if sparse:
            return tf.sparse_tensor_dense_matmul(x, y)
        return tf.matmul(x, y)

    def update_node_feat(self, node_feats, layer_idx=0):
        if self.GNN_model=='HATS':
            def to_input_shape(emb):
                    # [R,N,K,D]
                    emb_ = []
                    for i in range(emb.shape[0]):
                        exp = tf.tile(tf.expand_dims(tf.expand_dims(emb[i], 0),0),[self.num_companies, 20,1])
                        emb_.append(tf.expand_dims(exp,0))
                    return tf.concat(emb_,0)
            with tf.variable_scope('graph_ops'):
                node_feats = tf.concat([tf.zeros([1,node_feats.shape[1]]), node_feats], 0)
                neighbors = tf.nn.embedding_lookup(node_feats, self.rel_mat)
                exp_state = tf.expand_dims(tf.expand_dims(node_feats[1:], 1), 0)
                exp_state = tf.tile(exp_state, [self.num_relations, 1, 20, 1])
                rel_embs = to_input_shape(self.rel_emb)
                att_x = tf.concat([neighbors, exp_state, rel_embs], -1)
                score = tf.layers.dense(inputs=att_x, units=1, name='state_attention')
                att_mask_mat = tf.to_float(tf.expand_dims(tf.sequence_mask(self.rel_num, 20), -1))
                att_score = tf.nn.softmax(score, 2)
                all_rel_rep = tf.reduce_sum(neighbors*att_score, 2) / tf.expand_dims((tf.to_float(self.rel_num)+1e-10), -1)
                updated_state = tf.reduce_mean(all_rel_rep,0)
            return updated_state

    def simple_pooling(self, state, type):
        if type=='mean':
            pooled = tf.reduce_mean(state,0)
        elif type=='max':
            pooled = tf.reduce_max(state,0)
        return pooled

    def build_model(self):
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        # x [num company, lookback]
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.lookback, self.input_dim]) # (?, 50, 1) - input companies, lookback, feature dims
        self.y = tf.placeholder(tf.float32, shape=[None, 3]) # up, neutral, down                # (1, 3) - target company, classification output
        if self.GNN_model=='HATS':
            self.rel_mat = tf.placeholder(tf.int32, shape=[None, None, None]) # company, company
            self.rel_num = tf.placeholder(tf.int32, shape=[None, None]) # Edge, Node
            self.rel_emb = self.create_relation_onehot()

        state = self.get_state(price_model=self.config.price_model) # (n, d) - node, feature dims
        self.g_state = state[0]
        nodes_state = state[1:]

        self.nodes_state = self.update_node_feat(nodes_state)

        pooled = self.simple_pooling(nodes_state, 'max')
        graph_state = tf.expand_dims(tf.concat([self.g_state, pooled],0),0)

        logits = tf.layers.dense(inputs=graph_state, units=3, name='prediction', activation=tf.nn.leaky_relu)

        self.prob = tf.nn.softmax(logits)
        self.prediction = tf.argmax(logits, -1)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.cross_entropy,
                                                                                  global_step=self.global_step_tensor)

            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

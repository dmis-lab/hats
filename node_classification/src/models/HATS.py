import numpy as np
import os
import random
import re
import time

from base.base_model import BaseModel
import tensorflow as tf


class HATS(BaseModel):
    def __init__(self, config):
        super(HATS, self).__init__(config)
        self.n_labels = len(config.label_proportion)
        self.input_dim = len(config.feature_list)
        self.num_layer = config.num_layer
        self.keep_prob = 1-config.dropout
        self.max_grad_norm = config.grad_max_norm
        self.num_relations = config.num_relations
        self.node_feat_size = config.node_feat_size
        self.rel_projection = config.rel_projection
        self.feat_attention = config.feat_att
        self.rel_attention = config.rel_att
        self.att_topk = config.att_topk
        self.num_companies = config.num_companies
        self.neighbors_sample = config.neighbors_sample


        self.build_model()
        self.init_saver()

    def get_state(self, state_module):
        if state_module == 'lstm':
            cells = [tf.contrib.rnn.BasicLSTMCell(self.node_feat_size) for _ in range(1)]
            dropout = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob,
                                        output_keep_prob=self.keep_prob) for cell in cells]
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(dropout, state_is_tuple=True)
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.float32)
            state = tf.concat([tf.zeros([1,state[-1][-1].shape[1]]), state[-1][-1]], 0) # zero padding

        return state

    def relation_projection(self, state, rel_idx):
        with tf.variable_scope('relation_'+str(rel_idx)):
            rel_state = tf.layers.dense(inputs=state, units=self.node_feat_size,
                                activation=tf.nn.leaky_relu, name='projection')
        return rel_state

    def node_level_attention(self, node_feats, rel_mat, rel_num, rel_idx):
        with tf.variable_scope('node_attention_'+str(rel_idx)):
            # Neighbors [N, max_neighbors, Node feat dim]
            neighbors = tf.nn.embedding_lookup(node_feats, rel_mat)
            mask_mat = tf.to_float(tf.expand_dims(tf.sequence_mask(rel_num, self.max_k), -1))
            exp_node_feats = tf.tile(tf.expand_dims(node_feats[1:], 1), [1, self.max_k, 1])
            att_x = tf.concat([neighbors, exp_node_feats], -1)
        return att_x

    def get_relation_rep(self, state):
        # input state [Node, Original Feat Dims]
        with tf.variable_scope('graph_ops'):
            if self.feat_attention:
                neighbors = tf.nn.embedding_lookup(state, self.rel_mat)
                # exp_state [1, Nodes, 1, Feat dims]
                exp_state = tf.expand_dims(tf.expand_dims(state[1:], 1), 0)
                exp_state = tf.tile(exp_state, [self.num_relations, 1, self.max_k, 1])
                rel_embs = self.to_input_shape(self.rel_emb)

                # Concatenated (Neightbors with state) :  [Num Relations, Nodes, Num Max Neighbors, 2*Feat Dims]
                att_x = tf.concat([neighbors, exp_state, rel_embs], -1)

                score = tf.layers.dense(inputs=att_x, units=1, name='state_attention')
                att_mask_mat = tf.to_float(tf.expand_dims(tf.sequence_mask(self.rel_num, self.max_k), -1))
                att_score = tf.nn.softmax(score, 2)
                all_rel_rep = tf.reduce_sum(neighbors*att_score, 2) / tf.expand_dims((tf.to_float(self.rel_num)+1e-10), -1)

        return all_rel_rep

    def get_relations_rep(self, state):
        # old version without projection code
        with tf.name_scope('graph_ops'):
            # Neighbors : [Num Relations, Nodes, Num Max Neighbors, Feat Dims]
            neighbors = tf.nn.embedding_lookup(state, self.rel_mat)
            mask_mat = tf.to_float(tf.expand_dims(tf.sequence_mask(self.rel_num, self.max_k), -1))

            if self.feat_attention:
                # exp_state [1, Nodes, 1, Feat dims]
                exp_state = tf.expand_dims(tf.expand_dims(state[1:], 1), 0)
                exp_state = tf.tile(exp_state, [self.num_relations, 1, self.max_k, 1])

                # Concatenated (Neightbors with state) :  [Num Relations, Nodes, Num Max Neighbors, 2*Feat Dims]
                att_x = tf.concat([neighbors, exp_state], -1)
                score = tf.layers.dense(inputs=att_x, units=1, name='state_attention')
                att_score = tf.nn.softmax(score, 2)
                rel_rep = tf.reduce_sum(neighbors*att_score, 2) / tf.expand_dims((tf.to_float(self.rel_num)+1e-10), -1)
            else:
                rel_rep = tf.reduce_sum(neighbors, 2) / tf.expand_dims((tf.to_float(self.rel_num)+1e-10), -1)
        return rel_rep

    def aggregate_relation_reps(self,):
        def to_input_shape(emb):
            # [R,N,K,D]
            emb_ = []
            for i in range(emb.shape[0]):
                exp = tf.tile(tf.expand_dims(emb[i], 0),[self.num_companies,1])
                emb_.append(tf.expand_dims(exp,0))
            return tf.concat(emb_,0)
        with tf.name_scope('aggregate_ops'):
            # all_rel_rep : [Num Relations, Nodes, Feat dims]
            if self.rel_attention:
                rel_emb = to_input_shape(self.rel_emb)
                att_x = tf.concat([self.all_rel_rep,rel_emb],-1)
                att_score = tf.nn.softmax(tf.layers.dense(inputs=att_x, units=1,
                                        name='relation_attention'), 1)
                updated_state = tf.reduce_mean(self.all_rel_rep * att_score, 0)
            else:
                updated_state = tf.reduce_mean(self.all_rel_rep, 0)
        return updated_state

    def create_relation_embedding(self, ):
        return tf.get_variable("Relation_embeddings", [self.num_relations, 32])

    def create_relation_onehot(self, ):
        one_hots = []
        for rel_idx in range(self.num_relations):
            one_hots.append(tf.one_hot([rel_idx],depth=self.num_relations))
        return tf.concat(one_hots,0)
    
    def to_input_shape(self, emb):
        emb_ = []
        for i in range(emb.shape[0]):
            exp = tf.tile(tf.expand_dims(tf.expand_dims(emb[i], 0),0),[self.num_companies, self.neighbors_sample,1])
            emb_.append(tf.expand_dims(exp,0))
        return tf.concat(emb_,0)

    def build_model(self):
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        # x [num company, lookback]
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.lookback, self.input_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_labels])
        self.rel_mat = tf.placeholder(tf.int32, shape=[None, None, self.neighbors_sample]) # Edge, Node, Node
        self.rel_num = tf.placeholder(tf.int32, shape=[None, None]) # Edge, Node
        self.max_k = tf.placeholder(tf.int32, shape=())

        self.rel_emb = self.create_relation_onehot()

        self.exppanded = self.to_input_shape(self.rel_emb)

        state = self.get_state('lstm')
        # Graph operation
        self.all_rel_rep = self.get_relation_rep(state)

        # [Node, Feat dims]
        rel_summary = self.aggregate_relation_reps()
        updated_state = rel_summary+state[1:]

        logits = tf.layers.dense(inputs=updated_state, units=self.n_labels,
                                activation=tf.nn.leaky_relu, name='prediction')

        self.prob = tf.nn.softmax(logits)
        self.prediction = tf.argmax(logits, -1)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = self.cross_entropy
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(loss,
                                                                      global_step=self.global_step_tensor)

            correct_prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(self.y, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

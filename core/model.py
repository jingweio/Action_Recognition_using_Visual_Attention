# update: 8.14.2017
from __future__ import division
import tensorflow as tf


class CaptionGenerator(object):
    def __init__(self, label_to_idx, dim_feature, dim_hidden=1024, n_time_step=17,
                 ctx2out=True, alpha_c=1.0, selector=False, dropout=False):
        """
        Args:
            ctx2out: context to hidden state.
            alpha_c: Doubly stochastic regularization coefficient.
            selector: gating scalar for context vector.
            dropout: If true then dropout layer is added.
            V: the length of the possible labels
            L: the features' number of each image
            D: the features' dimension
            T: time step
            n_time_step: the same with T
        """
        # dictionary data
        self.label_to_idx = label_to_idx
        comp = lambda x, y: 1 if label_to_idx[x] > label_to_idx[y] else -1 if label_to_idx[x] < label_to_idx[y] else 0
        self.idx_to_label = sorted(label_to_idx.keys(), cmp=comp)

        # optional choice
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout

        # key parameters
        self.V = len(self.idx_to_label)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.H = dim_hidden
        # self.M = len(label_to_idx.keys())
        self.T = n_time_step
        self.n_time_step = n_time_step

        # initializer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.n_time_step, self.L, self.D])
        # labels contains label for each video in the batch-video
        self.label_idxs = tf.placeholder(tf.int32, [None, self.n_time_step])

    # set the initial state of lstm cell
    def _get_initial_lstm(self, features, name):
        with tf.variable_scope(name):
            features_mean = tf.reduce_mean(features, 1)
            features_mean = tf.reshape(features_mean, [-1, self.T, self.D])
            features_mean = tf.reduce_mean(features_mean, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    # reshape features? why do that projection? --- problem: the reason of doing that
    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    # attention model layer
    def _attention_layer(self, features, features_proj, h, reuse):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')
            return context, alpha

    # update the memory using selector
    def _selector(self, context, h, reuse):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.mul(beta, context, name='selected_context')
            return context, beta

    # decoding of lstm
    def _decode_lstm(self, h, context, dropout, reuse):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.V], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.V], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.V, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.V], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out

            return out_logits

    # normalize batch data
    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def build_model(self):
        # data prior process
        features = self.features
        features = tf.reshape(features, [-1, self.L, self.D])
        batch_size = tf.shape(features)[0] / self.V
        features = self._batch_norm(features, mode='test&val', name='Conv_features')
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []

        # LSTM ----------------------one layer---------------------------- START
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        c, h = self._get_initial_lstm(features=features, name='lstm-cell')
        # labels to label_idxs
        for t in range(self.T):  # each t of lstm-layers process
            context, alpha = self._attention_layer(features[t::self.T], features_proj[t::self.T], h,
                                                   reuse=(t != 0))
            alpha_list.append(alpha)

            # update after adding this procession the outcome improved dramatically --- more quick to learn that
            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope("LSTM", reuse=(t != 0)):
                _, (c, h) = lstm_cell(context, (c, h))

            logits = self._decode_lstm(h, context, dropout=self.dropout, reuse=(t != 0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.label_idxs[:, t]))
        # LSTM ----------------------one layer----------------------------- END

        # add the regulation to the cost function or not and the coefficients here needed to be adjusted
        if self.alpha_c > 0:  # # -------------------check it later
            alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))
            alphas_all = tf.reduce_sum(alphas, 1)
            alpha_reg = self.alpha_c * tf.reduce_sum((self.T / 196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self):
        # data collection
        features = self.features
        features = tf.reshape(features, [-1, self.L, self.D])
        features = self._batch_norm(features, mode='test&val', name='Conv_features')
        features_proj = self._project_features(features=features)

        alpha_list = []
        beta_list = []
        sampled_label_index_list = []

        # LSTM -----------------------one layer-------------------------------- START
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        c, h = self._get_initial_lstm(features=features, name='lstm-cell')
        for t in range(self.T):  # each t of lstm-layers process
            context, alpha = self._attention_layer(features[t::self.T], features_proj[t::self.T], h,
                                                   reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope("LSTM", reuse=(t != 0)):
                _, (c, h) = lstm_cell(context, (c, h))

            logits = self._decode_lstm(h, context, dropout=self.dropout, reuse=(t != 0))
        # LSTM ------------------------one layer------------------------------- END

            # logits to possibility
            possibility = tf.nn.softmax(logits)
            sampled_label_index_each_t = tf.argmax(possibility, 1)
            sampled_label_index_list.append(sampled_label_index_each_t)

        # alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
        # betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        alphas = alpha_list
        betas = beta_list
        return alphas, betas, sampled_label_index_list

# update: 8.14.2017
from __future__ import print_function
import tensorflow as tf
from utils import *
import numpy as np
import time
from datetime import datetime
import os


class CaptioningSolver(object):
    def __init__(self, model, data, **kwargs):

        self.model = model
        self.train_data = data['train_data']
        self.n_epochs = kwargs.pop('n_epochs')
        self.batch_size = kwargs.pop('batch_size')
        self.update_rule = kwargs.pop('update_rule')
        self.learning_rate = kwargs.pop('learning_rate')

        self.print_every = kwargs.pop('print_every')
        self.save_every = kwargs.pop('save_every')

        self.data_path = kwargs.pop('data_path')

        self.log_path = kwargs.pop('log_path')
        self.model_path = kwargs.pop('model_path')
        self.test_result_save_path = kwargs.pop('test_result_save_path')
        self.models_val_disp = kwargs.pop('models_val_disp')

        self.pretrained_model = kwargs.pop('pretrained_model')
        self.test_model = kwargs.pop('test_model')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        # create necessary folders
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.test_result_save_path):
            os.makedirs(self.test_result_save_path)

    def train(self):
        # train dataset
        features = np.array(self.train_data['features'])
        labels = np.array(self.train_data['labels'])
        video_ids = np.array(self.train_data['video_ids'])

        n_iters_per_epoch = int(len(labels) / self.batch_size)

        # # mix up the training data
        # rand_idxs = np.random.permutation(len(labels))
        # labels = labels1[rand_idxs]
        # video_ids = video_ids1[rand_idxs]
        # features = features[rand_idxs]

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        tf.get_variable_scope().reuse_variables()  # set the 'reuse' of each variable as True
        _, _, sam_labels = self.model.build_sampler()

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            tf.scalar_summary('batch_loss', loss)  # add those to the observing process

        # add the variables into observation
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads_and_vars:
            tf.histogram_summary(var.op.name + '/gradient', grad)

        # train_summary = tf.merge_all_summaries()
        print("The number of epoch: %d" % self.n_epochs)
        print("Data size: %d" % (len(labels)))
        print("Batch size: %d" % self.batch_size)
        print("Iterations per epoch: %d" % (n_iters_per_epoch))

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            print('Session created')
            tf.initialize_all_variables().run()
            # train_summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=50)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            # training process -- n_epochs iterations + n_iter_per_epoch
            for e in range(self.n_epochs):
                for i in range(n_iters_per_epoch):
                    S = i * self.batch_size
                    E = (i + 1) * self.batch_size
                    labels_batch = labels[S:E]
                    video_ids_batch = video_ids[S:E]
                    features_batch = features[S:E]
                    label_batch_idxs = np.array([[self.model.label_to_idx[per] for per in PER] for PER in labels_batch])

                    feed_dict = {self.model.features: features_batch, self.model.label_idxs: label_batch_idxs}
                    _, l = sess.run([train_op, loss], feed_dict)

                    curr_loss += l

                    # # write summary for tensorboard visualization
                    # if i % 10 == 0:
                    #     summary = sess.run(train_summary, feed_dict)
                    #     train_summary_writer.add_summary(summary, e * n_iters_per_epoch + i)

                    # show the current training condition
                    if (i + 1) % self.print_every == 0:
                        print("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, l))
                        sam_labels_list = sess.run(sam_labels, feed_dict)

                        # decode the training result
                        gen_label_idxs, gen_labels = decode(sam_labels_list, self.model.idx_to_label)
                        org_label_idxs = label_batch_idxs[:, 0]

                        # visualize the current comparison
                        for j in range(len(org_label_idxs)):
                            print(video_ids_batch[j])
                            Ground_truth = 'org: ' + str(org_label_idxs[j])
                            Generated_one = 'gen: ' + str(gen_label_idxs[j])
                            print(Ground_truth + '--V.S.--' + Generated_one)
                        print('the current accurancy rate: ' +
                              str(accurate_percentage(gen_label_idxs, org_label_idxs)))

                print("Previous epoch loss: ", prev_loss)
                print("Current epoch loss: ", curr_loss)
                print("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0

                # save model's parameters
                if (e + 1) % self.save_every == 0:
                    saver.save(sess, self.model_path + 'model', global_step=e + 1)
                    print("model-%s saved." % (e + 1))

    # test one model in test set
    def test(self):
        # train dataset
        test_data = load_data(self.data_path, 'test')
        features = np.array(test_data['features'])
        labels = np.array(test_data['labels'])
        video_ids = np.array(test_data['video_ids'])
        n_iterations = int(len(labels) / self.batch_size)

        test_result_save_path = self.test_result_save_path

        # build a graph to sample labels
        alphas, betas, sam_labels_test = self.model.build_sampler()

        # percentage record.txt
        percentage_txt = open(test_result_save_path + 'percentage record.txt', 'a')
        percentage_txt.write(str(datetime.now()) + '_test_' + '\r\n')
        percentage_txt.write('model path: ' + self.test_model + '\r\n')
        percentage_txt.write('\r\n')

        # detail record.txt
        txt_file = open(test_result_save_path + 'detailed_record.txt', 'a')
        txt_file.write(str(datetime.now()) + '_test_' + '\r\n')
        txt_file.write('model path: ' + self.test_model + '\r\n')
        txt_file.write('\r\n')

        MATCH_RESULT = {}
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            AP_ALL = np.ndarray(n_iterations)

            for iter in range(n_iterations):
                S = iter * self.batch_size
                E = (iter + 1) * self.batch_size

                # batch-data
                features_batch = features[S:E]
                labels_batch = labels[S:E]
                video_id_batch = video_ids[S:E]

                # run the model
                label_idxs = np.array([[self.model.label_to_idx[per] for per in PER] for PER in labels_batch])
                feed_dict = {self.model.features: features_batch, self.model.label_idxs: label_idxs}
                alps, bts, sam_label_list_test = sess.run([alphas, betas, sam_labels_test], feed_dict)

                # decode the obtained result
                gen_idxs, gen_labels = decode(sam_label_list_test, self.model.idx_to_label)
                org_idxs = label_idxs[:, 0]

                # show the result
                print('-------------------------------------------------------------')
                # write the compared result into a file
                for i in range(len(labels_batch)):
                    txt_file.write(video_id_batch[i] + '-- org_label: ' + str(org_idxs[i]) + '--V.S.-- gen_label: '
                                   + str(gen_idxs[i]) + '\n')

                    # save the match result
                    MATCH_RESULT[video_id_batch[i]] = gen_idxs[i] - org_idxs[i]

                AP = accurate_percentage(gen_idxs, org_idxs)
                AP_ALL[iter] = float(AP)
                print(str(iter) + ' batch -- accurate percentage: ' + str(AP))
                percentage_txt.write(str(iter) + ' batch -- accurate percentage: ' + str(AP) + '\r\n')

            # closed writing of percentage_txt
            percentage_txt.write('\r\n')
            percentage_txt.write('accuracy: ' + str(np.mean(AP_ALL)) + '\r\n')
            percentage_txt.write('\r\n')
            percentage_txt.close()

            # closed writing of txt_file
            txt_file.write('\r\n')
            txt_file.write('accuracy: ' + str(np.mean(AP_ALL)) + '\r\n')
            txt_file.write('\r\n')
            txt_file.close()

            MATCH_RESULT['AP_ALL'] = AP_ALL
            AP_ALL.astype(np.float64)
            print('The total accurate percentage is ' + str(np.mean(AP_ALL)) + '\r\n')
            hickle.dump(MATCH_RESULT, test_result_save_path + 'MATCH_RESULT.hkl')

            # return np.mean(AP_ALL)

    # test all the model in validation set
    def all_model_val(self):
        txt = open(self.models_val_disp, 'a')
        txt.write(str(datetime.now()) + '\r\n')
        txt.write('\r\n')

        models_path = self.model_path
        models = [per for per in list(os.walk(models_path))[0][-1] if per != 'checkpoint' and per[-4::] != 'meta']
        models = sorted(models, cmp=model_comp)

        val_data = load_data(self.data_path, 'val')
        features = np.array(val_data['features'])
        labels = np.array(val_data['labels'])
        n_iterations = int(len(labels) / self.batch_size)

        # build a graph to sample labels
        alphas, betas, sam_labels_test = self.model.build_sampler()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            # test each model in validation data set
            for cur_model in models:
                saver.restore(sess, self.model_path + cur_model)
                AP_ALL = 0
                for iter in range(n_iterations):
                    S = iter * self.batch_size
                    E = (iter + 1) * self.batch_size
                    # batch-data
                    features_batch = features[S:E]
                    labels_batch = labels[S:E]
                    # run the model
                    label_idxs = np.array([[self.model.label_to_idx[per] for per in PER] for PER in labels_batch])
                    feed_dict = {self.model.features: features_batch, self.model.label_idxs: label_idxs}
                    alps, bts, sam_label_list_test = sess.run([alphas, betas, sam_labels_test], feed_dict)
                    # decode the obtained result
                    gen_idxs, gen_labels = decode(sam_label_list_test, self.model.idx_to_label)
                    org_idxs = label_idxs[:, 0]
                    AP = accurate_percentage(gen_idxs, org_idxs)
                    AP_ALL += float(AP)
                print(cur_model + ': ' + str(round(AP_ALL / float(n_iterations), 4) * 100) + '%')
                txt.write(cur_model + ': ' + str(round(AP_ALL / float(n_iterations), 4) * 100) + '%' + '\r\n')
            txt.write('\r\n')

# update: 8.14.2017
import tensorflow as tf
from scipy import ndimage
from core.vggnet import Vgg19
from core.utils import *
import numpy as np
import os
import hickle
# from datetime import datetime


def comp(x, y):
    x_num = int(x[:-4])
    y_num = int(y[:-4])
    if x_num > y_num:
        return 1
    if x_num < y_num:
        return -1
    if x_num == y_num:
        return 0


def main():
    PATH = os.getcwd()
    vgg_model_path = PATH + '/data/imagenet-vgg-verydeep-19.mat'
    num_of_image_per_video = 17
    type = ['train', 'val', 'test']
    # TIME = str(datetime.now())
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for each in type:

            # settle down the paths
            path = PATH + '/data/data_set/' + each + '/'
            save_path_feats = path + 'features_' + each + '.hkl'
            save_path_labels_all = path + 'labels_all_' + each + '.hkl'

            # load video_filenames and labels
            video_filename = load_pickle(path + 'video_filenames_' + each + '.pkl')
            labels = load_pickle(path + 'labels_' + each + '.pkl')

            # gather the whole data in the current type
            all_feats = np.ndarray([len(video_filename), num_of_image_per_video, 196, 512], dtype=np.float32)
            all_labels = [None] * len(video_filename)

            # feature extraction
            for idx, vf in enumerate(video_filename):
                images_list = sorted(list(os.walk(vf))[0][-1], cmp=comp)
                print ('Processed' + str(idx + 1) + 'videos..')

                # # generate images_path
                cur_images_path = [vf + '/' + image for image in images_list]
                step = int(float(len(images_list)) / float(num_of_image_per_video))
                print(step)

                # Supplement
                if step == 0:
                    cur_images_path += [cur_images_path[-1]] * (num_of_image_per_video - len(cur_images_path))

                # do not jump
                if step == 1:
                    # cut from the middle
                    start_num = np.floor(float(len(images_list) - num_of_image_per_video) / 2)
                    start = 1 if start_num == 0 else start_num
                    cur_images_path = cur_images_path[int(start - 1):int(num_of_image_per_video + start - 1)]

                # jump
                if step > 1:
                    # cut by jumping --  start from the bottom of each partition
                    cur_images_path = cur_images_path[step - 1::step]
                    # cut from the middle again in case of the residual effects
                    start_num = np.floor(float(len(cur_images_path) - num_of_image_per_video) / 2)
                    start = 1 if start_num == 0 else start_num
                    cur_images_path = cur_images_path[int(start - 1):int(num_of_image_per_video + start - 1)]

                # in case of failure
                if len(cur_images_path) != num_of_image_per_video:
                    print('step: ' + str(step))
                    print('length of origianl images: ' + str(len(images_list)))
                    print('length of standard: ' + str(num_of_image_per_video))
                    print('length: ' + str(len(cur_images_path)))
                    print('errors occur..')
                    exit()

                cur_labels = labels[idx]

                # read images and extract features
                image_batch = np.array(
                    map(lambda x: ndimage.imread(x, mode='RGB'), cur_images_path)).astype(np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})

                all_feats[idx, :] = feats
                all_labels[idx] = [cur_labels] * num_of_image_per_video

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path_feats)
            all_labels = np.array(all_labels)
            hickle.dump(all_labels, save_path_labels_all)
            print ("Saved %s.." % save_path_feats)

            # # log each process
            # txt = open(path + 'log_' + each + '.txt', 'a')
            # txt.close()


main()

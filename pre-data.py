# update: 8.14.2017
from core.utils import *
import os
import subprocess
import numpy as np


# seprate the whole data into train, val, and test set
def seprate_data(folders, read_path, save_path):
    attributes = ['video_names', 'labels', 'video_filenames']
    sets = ['train', 'val', 'test']

    # build necessary variables
    for attr in attributes:
        for s in sets:
            exec (attr + '_' + s + '=[]')

    # process in each folder
    for each in folders:
        path = read_path + each + '/'

        video_names = load_pickle(path + each + '_video_names.pkl')
        label = load_pickle(path + each + '_label.pkl')
        number = len(video_names)

        Video_Filenames = [path + per + '/' for per in video_names]  # video_filenames
        Video_Names = [each + '_' + name for name in video_names]  # VideoIds
        Labels = [label] * number  # labels

        # inside_mix up -- different people's mixing up
        Labels, Video_Filenames, Video_Names = mix_up(Labels, Video_Filenames, Video_Names)

        # list_lization
        Labels = list(Labels)
        Video_Filenames = list(Video_Filenames)
        Video_Names = list(Video_Names)

        # 0.6:0.20:0.20 distributed on the sets including train, val, and test sets
        n1 = int(np.ceil(number * 0.6))
        n2 = int(np.ceil(number * 0.20))

        print(str(number) + ' ' + str(n1) + ' ' + str(n2))
        print('label: ' + label)

        # video names
        video_names_train += Video_Names[:n1]
        video_names_val += Video_Names[n1:n1 + n2]
        video_names_test += Video_Names[n1 + n2:]

        # labels
        labels_train += Labels[:n1]
        labels_val += Labels[n1:n1 + n2]
        labels_test += Labels[n1 + n2:]

        # video_filenames
        video_filenames_train += Video_Filenames[:n1]
        video_filenames_val += Video_Filenames[n1:n1 + n2]
        video_filenames_test += Video_Filenames[n1 + n2:]

    # train_mix up
    video_names_train, labels_train, video_filenames_train = mix_up(video_names_train, labels_train,
                                                                    video_filenames_train)
    # validation_mix up
    video_names_val, labels_val, video_filenames_val = mix_up(video_names_val, labels_val, video_filenames_val)
    # test_mix up
    video_names_test, labels_test, video_filenames_test = mix_up(video_names_test, labels_test, video_filenames_test)

    # create folders:
    if not os.path.exists(save_path + 'train/'):
        os.makedirs(save_path + 'train/')
    if not os.path.exists(save_path + 'test/'):
        os.makedirs(save_path + 'test/')
    if not os.path.exists(save_path + 'val/'):
        os.makedirs(save_path + 'val/')

    # save video_names
    save_pickle(video_names_train, save_path + 'train/' + 'video_names_train.pkl')
    save_pickle(video_names_val, save_path + 'val/' + 'video_names_val.pkl')
    save_pickle(video_names_test, save_path + 'test/' + 'video_names_test.pkl')
    # save labels
    save_pickle(labels_train, save_path + 'train/' + 'labels_train.pkl')
    save_pickle(labels_val, save_path + 'val/' + 'labels_val.pkl')
    save_pickle(labels_test, save_path + 'test/' + 'labels_test.pkl')
    # save video_filenames
    save_pickle(video_filenames_train, save_path + 'train/' + 'video_filenames_train.pkl')
    save_pickle(video_filenames_val, save_path + 'val/' + 'video_filenames_val.pkl')
    save_pickle(video_filenames_test, save_path + 'test/' + 'video_filenames_test.pkl')


def video2image(rpath, spath, n):
    # '-r ' + str(n) +
    strcmd = 'ffmpeg -i ' + '"' + rpath + '"' + ' -vframes ' + str(
        n) + ' -s 224*224 -f image2 ' + '"' + spath + '%d.jpg"'
    subprocess.call(strcmd, shell=True)


# def scan_folder(path):
#     return len(list(os.walk(path))[0][1])


# mix up the order of the items in x, y and z separately
def mix_up(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    n = len(x)
    rand_idxs = np.random.permutation(n)
    x = x[rand_idxs]
    y = y[rand_idxs]
    z = z[rand_idxs]
    return x, y, z


def main():
    video_path = '/home/jingwei/Action Detection/video-ucla-website/'
    data_path = '/home/jingwei/Action Detection/A-R/data/'
    folders = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a08', 'a09', 'a11', 'a12']

    # folder-type to label
    video_labels_dic = {'a01': 'pick up with one hand',
                        'a02': 'pick up with two hands',
                        'a03': 'drop trash',
                        'a04': 'walk around',
                        'a05': 'sit down',
                        'a06': 'stand up',
                        'a08': 'donning',
                        'a09': 'doffing',
                        'a11': 'throw',
                        'a12': 'carry'}
    for type in folders:
        cur_video_path = video_path + type + '/'
        cur_image_path = data_path + 'image/' + type + '/'

        # obtain the label of videos
        try:
            label = open(cur_video_path + 'label-' + type + '.txt').readline()[:-1]
        except:
            label = video_labels_dic[type]

        # build the folder of each type -- a01, a02, ..., a12 and save the label for each type
        if not os.path.exists(cur_image_path):
            os.makedirs(cur_image_path)
        save_pickle(label, cur_image_path + type + '_label.pkl')

        video_names = []
        # images_per_video = 17

        # read the name of videos
        video_txt = open(cur_video_path + 'videos.txt').readlines()

        # cut images from videos and resize them
        for index, name in enumerate(video_txt):
            print ('video' + name[:-1] + 'process ... ')

            # remove the data type
            name = name[:-5]
            rpath = cur_video_path + name + '.avi'
            spath = cur_image_path + name + '/'

            # build the folder for saved images
            if not os.path.exists(spath):
                os.makedirs(spath)

            # cut videos into images
            MAX = 100
            video2image(rpath, spath, MAX)
            video_names.append(name)

        # save the videos_names in images folder
        video_names = np.array(video_names)
        save_pickle(video_names, cur_image_path + '/' + type + '_video_names.pkl')

    # divide the data into train, val, and test
    seprate_data(folders, data_path + '/image/', data_path + '/data_set/')

    # label to idx dictionary
    label_to_idx = {'pick up with one hand': 1, 'pick up with two hands': 2, 'drop trash': 3, 'walk around': 4,
                    'sit down': 5, 'stand up': 6, 'donning': 7, 'doffing': 8, 'throw': 9, 'carry': 0}
    save_pickle(label_to_idx, data_path + '/data_set/label_to_idx.pkl')


main()

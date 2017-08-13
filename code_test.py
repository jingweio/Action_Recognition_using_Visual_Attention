from core.utils import *
import collections
import os
# #
# # path = os.getcwd() + '/data/'
# # folders = ['train', 'val', 'test']
# # for f in folders:
# #     curpath = path + '/' + f + '/'
# #     with open(curpath + 'data_list_' + f + '.txt', 'w') as txt:
# #         labels = load_pickle(curpath + '/labels_' + f + '.pkl')
# #         video_ids = load_pickle(curpath + '/video_ids_' + f + '.pkl')
# #         for j in range(len(labels)):
# #             txt.write('video_id: ' + video_ids[j] + '-------' + 'labels: ' + labels[j] + '\r\n')
# # frames = load_pickle('/home/clarence/Image Caption/A-R/fram_num.pkl')
# # f = []
# # for each in frames:
# #     print(collections.Counter(each).keys())
# # f = load_pickle('/home/clarence/Image Caption/A-R/data/val/features_val.pkl')
# # print(len(f[0]))
# # print(len(f))
# data = load_pickle('/home/clarence/Image Caption/A-R/data/train/video_filenames_train.pkl')
# print(len(data))
# import subprocess
#
# rpath = '/home/jingwei/2/1.avi'
# spath = '/home/jingwei/2/1/'
# strcmd = 'ffmpeg -i ' + '"' + rpath + '"' + ' -vframes 20 -f image2 ' + '"' + spath + '%d.jpg"'
# subprocess.call(strcmd, shell=True)

label = load_pickle('/home/jingwei/Action Detection/A-R/data/image/a03/a03_label.pkl')
print(label)
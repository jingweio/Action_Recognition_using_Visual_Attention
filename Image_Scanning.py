# update: 8.14.2017
import os
from datetime import datetime

comp = lambda x, y: 1 if int(x) > int(y) else -1 if int(x) < int(y) else 0
image_path = '/home/jingwei/Action Detection/A-R/data/image/'
folders = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a08', 'a09', 'a11', 'a12']

cur_txt = open(image_path + 'Video_distribution.txt', 'a+')
cur_txt.write(str(datetime.now()) + '\r\n')

for folder in folders:
    tmp_dic = {}
    cur_txt.write(folder + ': ' + '\r\n')
    cur_path = image_path + folder + '/'
    sub_folders_list = list(os.walk(cur_path))[0][1]
    valid_num = 0
    for sub_folder in sub_folders_list:
        cur_sub_path = cur_path + sub_folder + '/'
        images_list_length = len(list(os.walk(cur_sub_path))[0][-1])
        # dict_building
        if tmp_dic.keys().count(str(images_list_length)) == 0:
            tmp_dic[str(images_list_length)] = 1
        else:
            tmp_dic[str(images_list_length)] += 1
    tmp_dic_keys = sorted(tmp_dic.keys(), cmp=comp)
    # write txt
    for per in tmp_dic_keys:
        cur_txt.write(per + ' frames: ' + str(tmp_dic[per]) + ' videos\n')
    cur_txt.write('total number of videos: ' + str(len(sub_folders_list)) + '\n')
    cur_txt.write('\r\n')
    print(folder + ' processed..')
cur_txt.close()

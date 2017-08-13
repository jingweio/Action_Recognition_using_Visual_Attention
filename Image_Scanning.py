import os

comp = lambda x, y: 1 if int(x) > int(y) else -1 if int(x) < int(y) else 0
image_path = '/home/jingwei/Action Detection/A-R/data/image/'
folders = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a08', 'a09', 'a11', 'a12']

for folder in folders:
    tmp_dic = {}
    cur_path = image_path + folder + '/'
    cur_txt = open(image_path + folder + '.txt', 'a+')
    sub_folders_list = list(os.walk(cur_path))[0][1]
    for sub_folder in sub_folders_list:
        cur_sub_path = cur_path + sub_folder + '/'
        images_list_length = len(list(os.walk(cur_sub_path))[0][-1])
        if tmp_dic.keys().count(str(images_list_length)) == 0:
            tmp_dic[str(images_list_length)] = 1
        else:
            tmp_dic[str(images_list_length)] += 1
        cur_txt.write(sub_folder + ': ' + str(images_list_length) + '\n')
    tmp_dic_keys = sorted(tmp_dic.keys(), cmp=comp)
    for per in tmp_dic_keys:
        cur_txt.write(per + ': ' + str(tmp_dic[per]) + '\n')
    cur_txt.write('total number of videos: ' + str(len(sub_folders_list)) + '\n')
    cur_txt.close()
    print(folder + ' processed..')

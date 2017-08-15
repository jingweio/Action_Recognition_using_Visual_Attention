# update: 8.14.2017
from core.utils import *
from collections import Counter
from datetime import datetime

data_path = '/home/jingwei/Action Detection/A-R/data/data_set/'
folders = ['train', 'test', 'val']
label_to_id = load_pickle(data_path + 'label_to_idx.pkl')

txt = open(data_path + 'partition_distribution.txt', 'a+')
txt.write(str(datetime.now()) + '\r\n')

for folder in folders:
    txt.write(folder + '_partition_distribution:' + '\r\n')
    cur_labels = load_pickle(data_path + folder + '/' + 'labels_' + folder + '.pkl')
    # analysis_distribution
    label_ids = [label_to_id[per] for per in cur_labels]
    total = len(label_ids)
    dict_counter = Counter(label_ids)
    txt.write('Total number: ' + str(total) + '\r\n')

    for type in range(0, 10):
        percentage = float(dict_counter[type + 1]) / float(total)
        txt.write(str(type + 1) + ': ' + str(round(float(percentage), 4) * 100) + '%\r\n')
    txt.write('\r\n')
    print(folder + ' processed..')
txt.close()

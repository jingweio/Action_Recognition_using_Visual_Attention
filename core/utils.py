import numpy as np
import cPickle as pickle
import time
import collections
import hickle


# decode the result
def decode(gen_label_list, idx_to_label):
    N = len(gen_label_list[0])
    labels_video = [None] * N
    label_idxs_video = [None] * N
    for j in range(N):
        possible_results = [label_t[j] for label_t in gen_label_list]
        temp = collections.Counter(possible_results).most_common()[0][0]
        label_idxs_video[j] = temp
        labels_video[j] = idx_to_label[temp]
    return np.array(label_idxs_video), np.array(labels_video)


# calculate the AP
def accurate_percentage(x, y):
    isSame = x - y
    return float(sum([1 for each in isSame if each == 0])) / float(len(x))


# load the train, val, test data set
def load_data(data_path, split):
    start_t = time.time()
    features = hickle.load(data_path + split + '/' + 'features_' + split + '.hkl')
    labels = hickle.load(data_path + split + '/' + 'labels_all_' + split + '.hkl')
    video_ids = load_pickle(data_path + split + '/' + 'video_names_' + split + '.pkl')  # name == id
    video_filename = load_pickle(data_path + split + '/' + 'video_filenames_' + split + '.pkl')
    data = {'features': features, 'labels': labels, 'video_ids': video_ids, 'video_filenames': video_filename}
    end_t = time.time()
    print "Elapse time: %.2f" % (end_t - start_t)
    return data


# load pickle type data
def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' % path)
        return file


# save data in pickle
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' % path)

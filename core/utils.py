# update: 8.14.2017
import numpy as np
import cPickle as pickle
import time
import collections
import hickle
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


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
    return round(float(sum([1 for each in isSame if each == 0])) / float(len(x)), 4)


def model_comp(x, y):
    num1 = int(x[x.find('-') + 1:])
    num2 = int(y[y.find('-') + 1:])
    if num1 > num2:
        return 1
    if num1 < num2:
        return -1
    if num1 == num2:
        return 0


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


def Send_email(sender, user, theme, txt, password=".."):
    msg = MIMEText(txt, 'plain', 'utf-8')
    n = sender.find('@')
    m = user.find('@')
    msg['From'] = formataddr([sender[:n + 1], sender])
    msg['To'] = formataddr([user[:m + 1], user])
    msg['Subject'] = theme

    server = smtplib.SMTP("smtp-mail.***.com", 666)
    server.ehlo()
    server.starttls()
    server.login(sender, password)
    server.sendmail(sender, [user, ], msg.as_string())
    server.quit()

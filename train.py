# update: 8.14.2017
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import *

current_path = '/home/jingwei/Action Detection/A-R/data/'


def main():
    train_data = load_data(current_path + 'data_set/', 'test')
    length = len(train_data['video_ids'])
    train_data['features'] = train_data['features'][:int(0.7 * length)]
    train_data['labels'] = train_data['labels'][:int(0.7 * length)]
    train_data['video_ids'] = train_data['video_ids'][:int(0.7 * length)]
    train_data['video_filenames'] = train_data['video_filenames'][:int(0.7 * length)]

    # train_data = {}

    data = {'train_data': train_data}
    label_to_idx = load_pickle(current_path + 'data_set/label_to_idx.pkl')
    num_images_per_video = 17

    model = CaptionGenerator(label_to_idx=label_to_idx, dim_feature=[196, 512],
                             dim_hidden=1024, n_time_step=num_images_per_video, ctx2out=True,
                             alpha_c=1.0, selector=True, dropout=False)

    solver = CaptioningSolver(model, data, n_epochs=500, batch_size=15, update_rule='adam',
                              learning_rate=0.0006, print_every=3, save_every=10,
                              pretrained_model=None, model_path=current_path + 'model/lstm/',
                              test_model=current_path + 'model/lstm/model-430', log_path=current_path + 'log/',
                              data_path=current_path + '/data_set/',
                              test_result_save_path=current_path + 'data_set/test/model_test_result/',
                              models_val_disp=current_path + 'model/models_accuracy_val.txt')

    solver.train()
    solver.all_model_val()
    # solver.test()


if __name__ == "__main__":
    main()

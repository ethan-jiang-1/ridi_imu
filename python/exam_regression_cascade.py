import os
import numpy as np
# import pandas
from sklearn.model_selection import train_test_split

# hack to hook modules in subfolder
import os
import sys
app_root = os.path.dirname(os.path.dirname(__file__))
app_root_python = app_root + "/pyton"
if app_root not in sys.path:
    sys.path.append(app_root)
if app_root_python not in sys.path:
    sys.path.append(app_root_python)

from python.regression_cascade import SVRCascade, SVRCascadeOption
from python.regression_cascade import get_best_option, load_datalist
from python.regression_cascade import write_model_to_file
import python.training_data as td


class RcRunner(object):
    def __init__(self, args):
        self.args = args
    
        self.class_map = {}

        self.feature_train = None
        self.label_train = None
        self.responses_train = None
        self.feature_test = None
        self.label_test = None
        self.responses_test = None

        self.label_pred = None
        self.responses_pred = None

    def _run_regression(self):
        best_option = SVRCascadeOption()
        if self.args.option:
            best_option.load_from_file(self.args.option)
            print('Options loaded from file: ', self.args.option)
        else:
            print('No option file is provided, running grid search')
            best_option = get_best_option(self.feature_train, self.label_train, self.class_map, self.responses_train, n_split=args.cv)
        model = SVRCascade(best_option, self.class_map)
        print('Sample used for training: ', self.feature_train.shape[0])
        model.train(self.feature_train, self.label_train.astype(np.int32), self.responses_train)

        if self.args.output_path:
            write_model_to_file(self.args.output_path, model)

        self.label_pred, self.responses_pred = None, None
        if self.label_test.shape[0] > 0:
            print('Running trained model on testing set:', self.feature_test.shape[0])
            self.label_pred, self.responses_pred = model.test(self.feature_test, self.label_test.astype(np.int32), self.responses_test)

        self.model = model


    def _check_result(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))
        plt.subplot(211)
        plt.plot(self.responses_test[0], label="test0", color='r')
        plt.plot(self.responses_pred[0], label="pred0", color='g')
        plt.legend()        
        plt.subplot(212)
        plt.plot(self.responses_test[1], label="test1", color='b')
        plt.plot(self.responses_pred[1], label="pred1", color='k')
        plt.legend()
        plt.show()

    def _load_dataset(self):
        if not self.args.list:
            raise ValueError("args.list need to be specified")

        option = td.TrainingDataOption()
        option.sample_step_ = self.args.step_size
        self.feature_all, self.label_all, self.responses_all, self.class_map = load_datalist(path=self.args.list, option=option)
        print("loaded raw data:")
        print("feature_all: ", self.feature_all.shape)
        print("response_all:", self.responses_all.shape)

        self.responses_all = self.responses_all[:, [0, 2]]

        print('Data loaded. Total number of samples: ', self.feature_all.shape[0])

        for key, value in self.class_map.items():
            print('%d samples in %s(label %d)' % (len(self.label_all[self.label_all == value]), key, value))


    def _split_dataset(self):

        # Combine label and response to a single array to simplify the splitting process.
        target_temp = np.concatenate([self.label_all[:, None], self.responses_all], axis=1)
        self.feature_train, self.feature_test, self.target_train, self.target_test = train_test_split(self.feature_all, target_temp,
                                                                                train_size=self.args.train_ratio)

        print('Randomly splitting the dataset to %d/%d samples for training/testing.' % (self.feature_train.shape[0], self.feature_test.shape[0]))
        self.label_train, self.responses_train = self.target_train[:, 0], self.target_train[:, 1:]
        self.label_test, self.responses_test = self.target_test[:, 0], self.target_test[:, 1:]


        if self.args.train_test_path:
            self._np_save_dataset()

        if self.args.subsample > 1:
            print("subsample: ", self.args.subsample)
            self.feature_train = self.feature_train[0:-1: self.args.subsample]
            self.label_train = self.label_train[0:-1: self.args.subsample]
            self.responses_train = self.responses_train[0:-1: self.args.subsample]

    def _np_save_dataset(self):
        if not os.path.exists(self.args.train_test_path):
            os.makedirs(self.args.train_test_path)
        train_all = np.concatenate([self.feature_train, self.label_train[:, None], self.responses_train], axis=1)
        test_all = np.concatenate([self.feature_test, self.label_test[:, None], self.responses_test], axis=1)
        np.save(self.args.train_test_path + '/train.npy', train_all)
        np.save(self.args.train_test_path + '/test.npy', test_all)
        with open(self.args.train_test_path + '/class_map.txt', 'w') as f:
            f.write('%d\n' % len(self.class_map))
            for k, v in self.class_map.items():
                f.write('{:s} {:d}\n'.format(k, v))
        print('Training/testing set written to ' + self.args.train_test_path)


    def run(self):
        self._load_dataset()
        self._split_dataset()
        self._run_regression()
        self._check_result()


def _exec_regression_cascade(args):
    runner = RcRunner(args)
    runner.run()

    # load_from_list = True
    # data_loaded = False
    # feature_train, label_train, responses_train = None, None, None
    # feature_test, label_test, responses_test = None, None, None
    # class_map = {}
    # train_file_path, test_file_path = None, None

    # if args.train_test_path:
    #     train_file_path = args.train_test_path + '/train.npy'
    #     test_file_path = args.train_test_path + '/test.npy'
    #     if os.path.exists(train_file_path) and os.path.exists(test_file_path) and os.path.exists(
    #             args.train_test_path + '/class_map.txt'):
    #         print('Loading training set from ', train_file_path)
    #         train_all = np.load(train_file_path)
    #         print('Loading testing set from ', test_file_path)
    #         test_all = np.load(test_file_path)
    #         feature_train, label_train, responses_train = train_all[:,:-3], train_all[:, -3], train_all[:, -2:]
    #         feature_test, label_test, responses_test = test_all[:,:-3], test_all[:, -3], test_all[:, -2:]
    #         with open(args.train_test_path + '/class_map.txt') as f:
    #             num_classes = int(f.readline().strip())
    #             for i in range(num_classes):
    #                 line = f.readline().strip().split()
    #                 class_map[line[0]] = int(line[1])
    #         load_from_list = False
    #         data_loaded = True

    # if load_from_list and args.list:
    #     option = td.TrainingDataOption()
    #     option.sample_step_ = args.step_size
    #     feature_all, label_all, responses_all, class_map = load_datalist(path=args.list, option=option)
    #     print("loaded raw data:")
    #     print("feature_all: ", feature_all.shape)
    #     print("response_all:", responses_all.shape)

    #     responses_all = responses_all[:, [0, 2]]

    #     print('Data loaded. Total number of samples: ', feature_all.shape[0])

    #     for key, value in class_map.items():
    #         print('%d samples in %s(label %d)' %(len(label_all[label_all == value]), key, value))

    #     # Combine label and response to a single array to simplify the splitting process.
    #     target_temp = np.concatenate([label_all[:, None], responses_all], axis=1)
    #     feature_train, feature_test, target_train, target_test = train_test_split(feature_all, target_temp,
    #                                                                               train_size=args.train_ratio)

    #     print('Randomly splitting the dataset to %d/%d samples for training/testing.' %(feature_train.shape[0], feature_test.shape[0]))
    #     label_train, responses_train = target_train[:, 0], target_train[:, 1:]
    #     label_test, responses_test = target_test[:, 0], target_test[:, 1:]
    #     data_loaded = True
    #     if args.train_test_path:
    #         if not os.path.exists(args.train_test_path):
    #             os.makedirs(args.train_test_path)
    #         train_all = np.concatenate([feature_train, label_train[:, None], responses_train], axis=1)
    #         test_all = np.concatenate([feature_test, label_test[:, None], responses_test], axis=1)
    #         np.save(train_file_path, train_all)
    #         np.save(test_file_path, test_all)
    #         with open(args.train_test_path + '/class_map.txt', 'w') as f:
    #             f.write('%d\n' % len(class_map))
    #             for k, v in class_map.items():
    #                 f.write('{:s} {:d}\n'.format(k, v))
    #         print('Training/testing set written to ' + args.train_test_path)
    # if not data_loaded:
    #     raise ValueError('Both data list and train/test directory are invalid')

    # if args.subsample > 1:
    #     feature_train = feature_train[0:-1:args.subsample]
    #     label_train = label_train[0:-1:args.subsample]
    #     responses_train = responses_train[0:-1:args.subsample]

    # model, label_pred, responses_pred = _run_regression(args, feature_train, label_train, class_map, responses_train, feature_test, label_test, responses_test)
    # _check_result(args, label_pred, responses_pred, label_test, responses_test)


def _fake_args(args):
    ds_folder = os.path.dirname(os.path.dirname(__file__)) + "/ds_rdii/"

    args.list = ds_folder + "xlist_train_ethan0.txt"
    args.train_ratio = 0.9

    return args

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--list', default=None)
    parser.add_argument('--train_test_path', default=None, help='If set, the computed feature vectors, labels and'
                                                                'responses will be cached.')
    parser.add_argument('--output_path', default=None, type=str, help='The path where learned model is written.')
    parser.add_argument('--subsample', default=1, type=int, help='If greater than 1, the training set will be'
                                                                 'down-sampled.')
    parser.add_argument('--step_size', default=10, type=int, help='Number of frame between two feature extractions.')
    parser.add_argument('--train_ratio', default=0.99999, type=float)
    parser.add_argument('--cv', default=3, type=int, help='Number of folds during cross-valication.')
    parser.add_argument('--option', default=None, type=str, help='(Optional) The path to a file containing the '
                                                                 'hyperparameters. If not provided, a grid search'
                                                                 ' will be performed to obtain the best hyperparameter.')
    args = parser.parse_args()
    args = _fake_args(args)
    _exec_regression_cascade(args)


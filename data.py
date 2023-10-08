import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms


def load_data(a_data_path):
    with open(a_data_path, 'rb') as f:
        a_dataset = np.load(f)
        print(a_data_path, a_dataset.shape)

    return a_dataset


class BioData(Dataset):
    def __init__(self, a_data_path, transformer):

        self.all_X = load_data(a_data_path)
        self.transformer = transformer

    def __len__(self):
        return len(self.all_X)

    def __getitem__(self, idx):
        X = self.all_X[idx][np.newaxis, :]
        X = self.transformer(X)

        return X.float()


class NP_to_Tensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)


def get_bio_dataset(a_data_path, b_data_path):
    transformer = transforms.Compose([NP_to_Tensor()])
    return BioData(a_data_path, b_data_path, transformer)


def get_interictal_data_path(dataset_prefix, training):
    train_1_data_path = dataset_prefix + \
        'patient1_interictal_train.npy' if training else dataset_prefix + \
        'patient1_interictal_test.npy'
    train_3_data_path = dataset_prefix + \
        'patient3_interictal_train.npy' if training else dataset_prefix + \
        'patient3_interictal_test.npy'
    train_5_data_path = dataset_prefix + \
        'patient5_interictal_train.npy' if training else dataset_prefix + \
        'patient5_interictal_test.npy'
    train_2_data_path = dataset_prefix + \
        'patient2_interictal_train.npy' if training else dataset_prefix + \
        'patient2_interictal_test.npy'
    train_6_data_path = dataset_prefix + \
        'patient6_interictal_train.npy' if training else dataset_prefix + \
        'patient6_interictal_test.npy'
    train_10_data_path = dataset_prefix + \
        'patient10_interictal_train.npy' if training else dataset_prefix + \
        'patient10_interictal_test.npy'

    return train_1_data_path, train_3_data_path, train_5_data_path, train_2_data_path, train_6_data_path, train_10_data_path


def get_ictal_data_path(dataset_prefix):
    train_1_data_path = dataset_prefix + 'patient1_ictal.npy'
    train_3_data_path = dataset_prefix + 'patient3_ictal.npy'
    train_5_data_path = dataset_prefix + 'patient5_ictal.npy'
    train_2_data_path = dataset_prefix + 'patient2_ictal.npy'
    train_6_data_path = dataset_prefix + 'patient6_ictal.npy'
    train_10_data_path = dataset_prefix + 'patient10_ictal.npy'

    return train_1_data_path, train_3_data_path, train_5_data_path, train_2_data_path, train_6_data_path, train_10_data_path


def get_data_loader(dataset_prefix, batch_size, device, shuffle=True, preload_gpu=False, training=True, ictal=False):
    train_1_data_path, train_3_data_path, train_5_data_path, train_2_data_path, train_6_data_path, train_10_data_path = get_interictal_data_path(
        dataset_prefix, training) if not ictal else get_ictal_data_path(dataset_prefix)

    if preload_gpu:
        train_1_data = load_data(train_1_data_path)
        train_3_data = load_data(train_3_data_path)
        train_5_data = load_data(train_5_data_path)
        train_2_data = load_data(train_2_data_path)
        train_6_data = load_data(train_6_data_path)
        train_10_data = load_data(train_10_data_path)
        train_data = np.concatenate(
            [train_1_data, train_3_data, train_5_data, train_2_data, train_6_data, train_10_data], axis=0)
        print('train_data', train_data.shape)
        train_data = torch.from_numpy(
            train_data.copy()).float().to(device)

        conds = [[1, 0, 0, 0, 0, 0]] * len(train_1_data) + \
                [[0, 1, 0, 0, 0, 0]] * len(train_3_data) + \
                [[0, 0, 1, 0, 0, 0]] * len(train_5_data) + \
                [[0, 0, 0, 1, 0, 0]] * len(train_2_data) + \
                [[0, 0, 0, 0, 1, 0]] * len(train_6_data) + \
                [[0, 0, 0, 0, 0, 1]] * len(train_10_data)

        conds = np.array(conds)
        conds = torch.from_numpy(
            conds.copy()).float().to(device)

        train_cond_data = TensorDataset(train_data, conds)

        num_workers = 0
        pin_memory = False

    train_data_loader = DataLoader(
        train_cond_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    return train_data_loader

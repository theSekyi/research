import torch
import torchvision
import torchvision.transforms as transforms
import os
import shutil
import numpy as np
import pprint
from tabulate import tabulate
import pandas as pd
import pathlib
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder


def load_dataset(train_path, test_path, transformation):

    # Load all of the images, transforming them
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transformation)

    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transformation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def create_data_config(path, src, rounds, data_config, round_name="round"):
    folders = data_config.keys()
    for rnd in range(rounds):  # create folder for rounds
        round_folder = f"{path}/{round_name}_{rnd}"
        pathlib.Path(round_folder).mkdir(parents=True, exist_ok=True)
        for folder, data_size in data_config.items():  # create folder for labels
            dest = f"{path}/{round_name}_{rnd}/{folder}"
            label_src = f"{src}/{folder}"

            pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
            _, file_names = get_random_fls(label_src, num_files=data_size)
            mv_files(label_src, dest, file_names)
        dict_to_df(get_file_len(round_folder))


def get_file_len(path):

    if path.endswith("/"):  # check if path doesn't end in /
        path = path[:-1]

    iter_fl = iter(os.walk(path))
    dictn = {}
    next(iter_fl)
    for r, d, files in iter_fl:
        new_r = os.path.basename(r)

        if not new_r.startswith("."):  # ignore hidden directories ie.eg .ipython_checkpoints
            dictn[new_r] = len(files)
    return dictn


def dict_to_df(dictn):
    """Convert a dictionary to a nicely formatted table"""
    df = pd.DataFrame(dictn.items(), columns=["Directory", "Number of Files"])
    print(tabulate(df, headers=df.columns, tablefmt="fancy_grid", showindex="never"))


def get_random_fls(path, percentage=None, num_files=None):
    files = [
        os.path.join(str(path), f)
        for f in os.listdir(path)
        if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".JPEG", "jpeg"))
    ]
    if num_files:
        random_files = np.random.choice(files, int(num_files), replace=False)
    else:
        random_files = np.random.choice(files, int(len(files) * percentage), replace=False)
    fl_names = [x.split("/")[-1] for x in random_files]

    return random_files, fl_names


def get_combined_anomalies(path, num_files):
    import pathlib

    individual_file_size = int(num_files / 2)

    files = [p.name for p in pathlib.Path(path).iterdir() if p.is_file()]

    dogs = [x for x in files if x.startswith("dog")]
    horses = [x for x in files if x.startswith("horse")]

    random_horses = np.random.choice(horses, individual_file_size, replace=False)
    random_dogs = np.random.choice(dogs, individual_file_size, replace=False)

    combined_classes = random_horses.tolist() + random_dogs.tolist()

    return combined_classes


def group_files_by_labels(files, labels_to_combine):
    from collections import defaultdict

    grouped_files = defaultdict(list)

    for label in labels_to_combine:
        for fl in files:
            if fl.startswith(label):
                grouped_files[label].append(fl)
    return grouped_files


def get_individual_file_size(individual_file_size, len_labels):
    from random import randint

    random_label_index = randint(0, len_labels - 1)
    lst_individual_file_size = [individual_file_size] * len_labels
    lst_individual_file_size[random_label_index] = 3
    return lst_individual_file_size


def get_n_combined_classes(path, num_files):

    labels_to_combine = [
        "automobile",
        "ship",
        "dog",
        "cat",
        "horse",
        "frog",
        "deer",
        "truck",
    ]

    len_labels = len(labels_to_combine)
    files = [p.name for p in pathlib.Path(path).iterdir() if p.is_file()]

    grouped_files = group_files_by_labels(files, labels_to_combine)

    combined_classes = []
    individual_file_size = int(num_files / len_labels)
    len_labels = len(grouped_files)
    individual_file_size_lst = get_individual_file_size(individual_file_size, len_labels)

    for i, (key, value) in enumerate(grouped_files.items()):

        lst_files = np.random.choice(value, individual_file_size_lst[i], replace=False).tolist()
        combined_classes.extend(lst_files)

    return combined_classes


def mv_files(src, dest, fl_names):
    for fl in fl_names:
        shutil.move(os.path.join(src, fl), dest)


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path

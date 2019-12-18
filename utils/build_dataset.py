import os
import numpy as np
import math
import logging
from utils.data_utils import class2label
logging.basicConfig(level=logging.INFO)


def read_file(file_path):
    with open(file_path,'r') as f:
        lines = f.readlines()
        return lines


def split_to_seen_unseen(sdp,sdp_pos,label_idx,seen_classes):

    seen_sdp = []
    seen_label = []
    seen_sdp_pos = []
    unseen_sdp = []
    unseen_label = []
    unseen_sdp_pos = []
    for i,label in enumerate(label_idx):
        if label in seen_classes:
            seen_sdp.append(sdp[i])
            seen_sdp_pos.append(sdp_pos[i])
            seen_label.append(label)
        else:
            unseen_sdp.append(sdp[i])
            unseen_sdp_pos.append(sdp_pos[i])
            unseen_label.append(label)

    return seen_sdp,seen_sdp_pos,seen_label,unseen_sdp,unseen_sdp_pos,unseen_label


def write_file(file_path,list_line):
    with open(file_path,'w') as f:
        for line in list_line:
            f.write(line+'\n')


def build_dataset_with_ratio(file_dir, ratio, n_classes=19,train=True):

    sdp_path = os.path.join(file_dir, 'val', 'sdp.txt')
    sdp_pos_path = os.path.join(file_dir, 'val', 'sdp_pos.txt')
    label_path = os.path.join(file_dir, 'val', 'labels.txt')

    if train:
        sdp_path = os.path.join(file_dir,'train','sdp.txt')
        sdp_pos_path = os.path.join(file_dir, 'train', 'sdp_pos.txt')
        label_path = os.path.join(file_dir, 'train', 'labels.txt')

    sdp = read_file(sdp_path)
    label = read_file(label_path)
    sdp_pos = read_file(sdp_pos_path)

    assert len(sdp) == len(label)
    assert len(sdp) == len(sdp_pos)

    n_seen_classes = math.ceil(ratio * n_classes)
    seen_classes = list(range(n_seen_classes))
    unseen_class = list(range(n_seen_classes, n_classes))

    train_label_idx = [class2label[x.strip()] for x in label]

    seen_sdp,seen_sdp_pos,seen_label,unseen_sdp,\
        unseen_sdp_pos,unseen_label = split_to_seen_unseen(sdp,sdp_pos,train_label_idx,seen_classes)

    if train:
        n_sample = len(seen_sdp)
        type = 'Train'
    else:
        n_sample = len(seen_sdp) + len(unseen_sdp)
        type = 'Val'

    logging.info("Build dataset with {0:.2f} % classes, number of sample:{1}, type: {2}".format(ratio*100,n_sample,type))

    return seen_sdp,seen_sdp_pos,seen_label,unseen_sdp,unseen_sdp_pos,unseen_label


if __name__ == '__main__':

    file_dir = 'data/processed'
    ratio = 1.0
    build_dataset_with_ratio(file_dir,ratio,train=False)
import os
import json
from data_utils import class2label


def convert_data(train_sdp_path,train_label_path,val_sdp_path,val_label_path,out_data_path):

    print("Start Convert Data ...")

    list_sdp  = []
    list_label = []
    with open(train_sdp_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            list_sdp.append(line)
    with open(val_sdp_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            list_sdp.append(line)
    with open(train_label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label_idx = class2label[line]
            list_label.append(label_idx)
    with open(val_label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label_idx = class2label[line]
            list_label.append(label_idx)

    result_dict = dict()
    result_dict['X'] = list_sdp
    result_dict['y'] = list_label
    with open(out_data_path,'w') as f:
        json.dump(result_dict,f)
    print("Done !!")

if __name__ == '__main__':
    train_sdp_path = 'data/processed/train/sdp.txt'
    train_label_path = 'data/processed/train/labels.txt'
    val_sdp_path = 'data/processed/val/sdp.txt'
    val_label_path = 'data/processed/val/labels.txt'
    out_data_path = 'data_convert.json'
    convert_data(train_sdp_path,train_label_path,val_sdp_path,val_label_path,out_data_path)
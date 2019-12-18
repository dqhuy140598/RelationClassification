from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.build_vocab import *
from utils.data_utils import class2label
from utils.build_dataset import build_dataset_with_ratio
import math
vocab_path = "data/processed/vocab.txt"
pos_vocab_path = 'data/processed/pos_vocab.txt'
depend_vocab_path = 'data/processed/depend_vocab.txt'
vocab_dict, num_words = load_vocab_to_dict(vocab_path)
pos_vocab_dict, num_pos = load_pos_vocab_to_dict(pos_vocab_path)
depend_vocab_dict, num_depend = load_depend_vocab_to_dict(depend_vocab_path)


class RelationDataset(Dataset):

    def __init__(self, file_dir, max_length_sdp, ratio=1.0, train=True):
        """
        constructor of the relation dataset class
        @param file_dir: path to processed data directory
        @param max_length_sdp: max sequence length of shortest dependency path
        @param train: True if train data else False
        """

        self.train = train
        self.n_classes = 19
        self.ratio = ratio
        self.max_length_sdp = max_length_sdp

        data = build_dataset_with_ratio(file_dir,ratio,self.n_classes,train=self.train)

        self.seen_sdp = data[0]
        self.seen_sdp_pos = data[1]
        self.seen_label = data[2]

        if not self.train:
            if self.ratio != 1.0:
                self.unseen_sdp = data[3]
                self.unseen_sdp_pos = data[4]
                self.unseen_label = data[5]
            else:
                self.unseen_sdp = None
                self.unseen_sdp_pos = None
                self.unseen_label = None

            if self.unseen_sdp is not None:

                self.seen_sdp.extend(self.unseen_sdp)
                self.seen_sdp_pos.extend(self.unseen_sdp_pos)
                self.seen_label.extend(self.unseen_label)

    def convert_sdp_to_idx(self, sdp):
        """
        Convert shortest dependency path to list indexes
        @param sdp: the input shortest dependency path
        @return: list of integers denotes the indexes of shortest dependency path in words vocabulary
        """
        words = sdp.strip().split(" ")
        list_idx = []
        for word in words:
            if word in vocab_dict.keys():
                idx = vocab_dict[word]
            else:
                idx = vocab_dict["<UNK>"]
            list_idx.append(idx)
        assert len(list_idx) == len(words)
        return list_idx

    def convert_to_one_hot(self, label_idx):
        """
        convert the label index to one hot encoding
        @param label_idx: a integer denotes the label class
        @return: a vector one hot with shape (n_classes)
        """
        label_onehot = np.zeros(shape=(self.n_classes), dtype=np.int32)
        label_onehot[label_idx] = 1
        return label_onehot

    def convert_depend_to_idx(self, depend):
        """
        convert depend to list index of depend vocabulary
        @param depend: list depend tokens
        @return: list of integers denotes the indexes of each depend tokens in depend vocabulary
        """
        words = depend.strip().split(" ")
        list_idx = []
        for word in words:
            if word in depend_vocab_dict.keys():
                idx = depend_vocab_dict[word]
            else:
                idx = depend_vocab_dict["<UNK>"]
            list_idx.append(idx)
        assert len(list_idx) == len(words)
        return list_idx

    def pad_depend_to_max_length(self, depend_idx):
        """
        pad depend indexes to max length
        @param depend_idx: list of integers denotes the depend indexes in depend vocabulary
        @return: the padded depend indexes
        """
        pad_idx = depend_vocab_dict["<PAD>"]
        n_depend_pad = (self.max_length_sdp - len(depend_idx)) // 2
        depend_idx = np.pad(depend_idx, pad_width=(n_depend_pad, self.max_length_sdp - n_depend_pad - len(depend_idx)),
                            constant_values=(pad_idx, pad_idx))
        return depend_idx

    def pad_to_max_length(self, sdp_idx):
        """
        pad sdp indexes to max length
        @param sdp_idx: list of integers denotes the shortest dependency path indexes in words vocabulary
        @return: the padded sdp indexes
        """
        pad_idx = vocab_dict["<PAD>"]
        n_sdp_pad = (self.max_length_sdp - len(sdp_idx)) // 2
        sdp_idx = np.pad(sdp_idx, pad_width=(n_sdp_pad, self.max_length_sdp - n_sdp_pad - len(sdp_idx)),
                         constant_values=(pad_idx, pad_idx))
        return sdp_idx

    def pad_pos_to_max_length(self, sdp_pos_idx):
        """
        pad pos indexes to max length
        @param sdp_pos_idx: list of integers denotes the sdp path of speech tagging in pos vocabulary
        @return: the padded sdp pos indexes
        """
        pad_idx = pos_vocab_dict["<PAD>"]
        n_sdp_pos_pad = (self.max_length_sdp - len(sdp_pos_idx)) // 2
        sdp_pos_idx = np.pad(sdp_pos_idx,
                             pad_width=(n_sdp_pos_pad, self.max_length_sdp - n_sdp_pos_pad - len(sdp_pos_idx)),
                             constant_values=(pad_idx, pad_idx))
        return sdp_pos_idx

    def convert_label_to_idx(self, label):
        """
        convert string label to integer label
        @param label: string label
        @return: integer label
        """
        label = label.strip()
        label_idx = class2label[label]
        return label_idx

    def convert_sdp_pos_to_idx(self, sdp_pos):
        """
        convert spd part of speech tagging to indexes
        @param sdp_pos: a string denotes the part of speech tagging of one sdp
        @return: list of integers denotes the indexes of the spd part of speech tagging in pos vocabulary
        """
        sdp_pos = sdp_pos.strip().split(" ")
        list_idx = []
        for token in sdp_pos:
            if token in pos_vocab_dict.keys():
                idx = pos_vocab_dict[token]
            else:
                idx = pos_vocab_dict[token]
            list_idx.append(idx)
        assert len(sdp_pos) == len(list_idx)
        return list_idx

    def __len__(self):
        """
        return the length of the dataset
        @return: a integer denotes the length of the dataset
        """
        return len(self.seen_sdp)

    def __getitem__(self, idx):
        """
        Get one sample from the dataset
        @param idx: index of the sample
        @return: processed data (sdp_idx_pad, sdp_pos_idx_pad, depend_idx_pad, label_one_hot)
        """
        sdp = self.seen_sdp[idx]
        label = self.seen_label[idx]
        sdp_pos = self.seen_sdp_pos[idx]
        # depend = self.depends[idx]
        # print(sdp)
        # print(sdp_pos)

        sdp_idx = self.convert_sdp_to_idx(sdp)
        sdp_pos_idx = self.convert_sdp_pos_to_idx(sdp_pos)
        # depend_idx = self.convert_depend_to_idx(depend)
        assert len(sdp_idx) == len(sdp_pos_idx)
        # print(sdp_idx,sdp_pos_idx)
        sdp_idx_pad = self.pad_to_max_length(sdp_idx)
        sdp_pos_idx_pad = self.pad_pos_to_max_length(sdp_pos_idx)
        # depend_idx_pad = self.pad_depend_to_max_length(depend_idx)
        label_idx = label
        label_one_hot = self.convert_to_one_hot(label_idx)
        return sdp_idx_pad, sdp_pos_idx_pad, label_one_hot


if __name__ == '__main__':

    train_dataset = RelationDataset('data/processed', max_length_sdp=13, ratio=0.75, train=False)
    print(train_dataset.__len__())
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0)
    for batch in train_loader:
        print(batch[0], batch[1], batch[2])
        break
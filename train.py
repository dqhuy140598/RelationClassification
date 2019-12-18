from utils.build_vocab import load_word_embedding
from utils.build_vocab import load_vocab_to_dict, load_pos_vocab_to_dict
from utils.helper import *
from gensim.models import KeyedVectors
from model.cnn import CNN
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from dataset import RelationDataset
from dataset_ratio import RelationDataset
import argparse
import math


def train(params, pretrained_path, use_thresh, ratio):
    """
    Train the cnn model
    @param params: hyper parameters
    @return: none
    """
    model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
    vocab_path = "data/processed/vocab.txt"
    pos_vocab_path = 'data/processed/pos_vocab.txt'
    # depend_vocab_path = 'data/processed/depend_vocab.txt'
    # train_sentences_path = 'data/processed/train/sdp.txt'
    # train_label_path = 'data/processed/train/labels.txt'
    #
    # test_sentences_path = 'data/processed/val/sdp.txt'
    # test_label_path = 'data/processed/val/labels.txt'

    file_dir = 'data/processed'

    vocab_dict, num_words = load_vocab_to_dict(vocab_path)
    pos_vocab_dict, num_pos = load_pos_vocab_to_dict(pos_vocab_path)

    embedding, coverage = load_word_embedding(model, embedding_size=params['embedding_size'], vocab=vocab_dict,
                                              vocab_size=num_words)

    cnn_model = CNN(word_embeddings=embedding, pos_size=params['pos_size'], depend_size=params['depend_size'],
                    params=params)

    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=params['lr'])

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    train_dataset = RelationDataset(file_dir=file_dir, max_length_sdp=params['max_length'], ratio=ratio, train=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)

    test_dataset = RelationDataset(file_dir=file_dir, max_length_sdp=params['max_length'], ratio=ratio, train=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

    seen_classes = int(math.ceil(ratio * 19))

    if seen_classes != 19:

        tmp = list(range(seen_classes + 1))

    else:

        tmp = list(range(seen_classes))

    scheduler.step()

    # a running average object for loss
    loss_avg = 0

    train_batch_num = train_dataset.__len__() // params['batch_size']
    test_batch_num = test_dataset.__len__() // params['batch_size']

    # Use tqdm for progress bar
    for epoch in range(params['epochs']):
        print("epoch:{0}/{1}".format(epoch, params['epochs']))
        epoch_loss = []
        epoch_acc = []
        val_loss = []
        val_acc = []
        cnn_model.train()

        train_out_put = []

        train_labels = []

        for batch in train_loader:
            # fetch the next training batch
            sdp, sdp_pos, label = batch

            label = label.type(torch.FloatTensor)
            # compute model output and loss
            batch_output = cnn_model([sdp, sdp_pos])

            # print(torch.sigmoid(batch_output))

            loss = cnn_model.loss(batch_output, label)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(cnn_model.parameters(), params['clip_grad'])

            # performs updates using calculated gradients
            optimizer.step()

            epoch_loss.append(loss.item())

            # acc = cal_acc(batch_output,label)

            # pred = torch.argmax(batch_output,dim=1)

            # correct = torch.sum(label==pred)

            # acc = correct.double()/batch_output.size(0)

            # acc = cal_acc(label,batch_output)

            # epoch_acc.append(acc)

            train_out_put.append(torch.sigmoid(batch_output).detach().numpy())

            label_idx = torch.argmax(label, dim=-1)

            train_labels.append(label_idx.numpy())

        train_out_put = np.concatenate(train_out_put, axis=0)

        train_labels = np.concatenate(train_labels, axis=0)

        print(train_out_put.shape)

        print(train_labels.shape)

        # cal mean and standard deviation from the output of train data
        mu_stds = cal_thresh(train_out_put, train_labels)

        print("evaluate on test set ............")

        cnn_model.eval()

        output_labels = list()
        target_labels = list()

        for batch in test_loader:
            # fetch the next training batch
            sdp, sdp_pos, label = batch

            label = label.type(torch.FloatTensor)

            # compute model output and loss
            batch_output = cnn_model([sdp, sdp_pos])
            loss = cnn_model.loss(batch_output, label)

            # Calculate the predicted class using train probability mean and standard deviation
            pred_class = convert_output_to_class(batch_output, mu_stds, seen_classes, use_thresh)

            val_loss.append(loss.item())

            # acc = cal_acc(batch_output,label)

            # pred = torch.argmax(batch_output,dim=1)

            # correct = torch.sum(label==pred)

            # acc = correct.double()/batch_output.size(0)

            # acc = cal_acc(label,batch_output)

            label_class = torch.argmax(label, dim=-1).data.numpy().tolist()

            # val_acc.append(acc)

            output_labels.extend(pred_class)
            target_labels.extend(label_class)

        # print('epochs:{0}, loss:{1:.2f}, accuracy:{2:.2f},val_loss:{3:.2f},val_acc:{4:.2f}'.format(epoch,cal_mean(epoch_loss),cal_mean(epoch_acc),cal_mean(val_loss),cal_mean(val_acc)))

        # print(output_labels)

        print('epochs:{0}, loss:{1:.2f},val_loss:{2:.2f}'.format(epoch, cal_mean(epoch_loss), cal_mean(val_loss)))

        print("classification report:")

        print(classification_report(target_labels, output_labels, labels=tmp))

    torch.save(cnn_model.state_dict(), 'model.pth')


if __name__ == '__main__':
    params_config = 'config/params.json'
    params = parse_json(params_config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', help='your pretrain      ed word2vec path', required=True)
    parser.add_argument('--use_thresh', \
                        help='If False then threshold =0.5 else calculate threshold from output probability', \
                        default=False, type=bool)
    parser.add_argument('--ratio', type=float, help='ratio', default=1.0)
    args = parser.parse_args()
    # print(type(args.cal_thresh))
    train(params, args.pretrained, args.use_thresh, args.ratio)
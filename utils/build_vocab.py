import os
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

# from gensim.models import KeyedVectors
# # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)


def build_vocab_to_file_txt(train_sentence_path,val_sentence_path,out_vocab_path):
    """
    Build word vocabulary from the train sentence file and val sentence file and write to vocab file
    @param train_sentence_path: the train sentence file path
    @param val_sentence_path: the val sentence file path
    @param out_vocab_path: the vocab file path
    @return: None
    """
    logging.info("Read words from sentences file......")
    list_word = []
    with open(train_sentence_path,'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            words = line.strip().split(" ")
            for word in words:
                if word not in list_word:
                    list_word.append(word)
    with open(val_sentence_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            words = line.strip().split(" ")
            for word in words:
                if word not in list_word:
                    list_word.append(word)
    logging.info("Write words to vocabularies file......")
    with open(out_vocab_path,'w') as f:
        for word in list_word:
            f.write(word+"\n")
        f.write("<UNK>"+'\n')
        f.write("<PAD>"+'\n')


def load_vocab_to_dict(vocab_path):
    """
    load words vocabulary to dictionary
    @param vocab_path: the word vocabulary file path
    @return: words vocabulary dictionary and num words
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError("Your Vocabulary File Path Not Found !!")
    with open(vocab_path,'r') as f:
        words = f.readlines()
        vocab_dict = dict()
        num_words = len(words)
        for i,word in enumerate(words):
            word = word.strip()
            vocab_dict[word] = i
        return vocab_dict,num_words


def load_word_embedding(embeddings,embedding_size,vocab,vocab_size):
    """
    load word embedding from pretrained word2vec
    @param embeddings: the gensim model word2vec
    @param embedding_size: the embeddings dim (300)
    @param vocab: words vocabulary dictionary
    @param vocab_size: words vocabulary size
    @return: embeddings matrix and the coverage of the words vocabulary and pretrained word2vec
    """
    init_w = np.random.rand(vocab_size,embedding_size)/np.sqrt(vocab_size)
    count=0
    for key,value in vocab.items():
      try:
        init_w[value] = embeddings.get_vector(key)
      except:
        count+=1
    pad_idx = vocab["<PAD>"]
    init_w[pad_idx] = np.zeros(shape=(1,embedding_size),dtype=np.float32)
    coverage = 1- float(count/vocab_size)
    return init_w,coverage


def load_pos_vocab_to_dict(pos_vocab_path):
    """
    load part of speech tagging words vocabulary to dictionary
    @param pos_vocab_path: part of speech tagging file path
    @return: part of speech tagging words vocabulary dictionary and num pos
    """
    if not os.path.exists(pos_vocab_path):
        raise FileNotFoundError("Your POS Vocabulary File Path Not Found !!")
    with open(pos_vocab_path,'r') as f:
        pos_tags = f.readlines()
        pos_vocab_dict = dict()
        num_pos = len(pos_tags)
        for i,pos in enumerate(pos_tags):
            pos = pos.strip()
            pos_vocab_dict[pos] = i
        return pos_vocab_dict,num_pos


def load_pos_embedding(pos_vocab,pos_size):
    """
    generate part of speech tagging embeddings matrix (one hot encoding)
    @param pos_vocab: part of speech tagging vocabulary dictionary
    @param pos_size: num of pos
    @return: embeddings matrix
    """
    init_w = np.eye(pos_size)
    pad_idx = pos_vocab["<PAD>"]
    init_w[pad_idx] = np.zeros(shape=(1,pos_size),dtype=np.float32)
    return init_w


def load_depend_vocab_to_dict(depend_vocab_path):
    """
    load depend vocabulary to dictionary
    @param depend_vocab_path: depend file path
    @return: depend dictionary
    """
    if not os.path.exists(depend_vocab_path):
        raise FileNotFoundError("Your POS Vocabulary File Path Not Found !!")
    with open(depend_vocab_path,'r') as f:
        depends = f.readlines()
        depend_dict = dict()
        num_depend = len(depends)
        for i,dep in enumerate(depends):
            dep = dep.strip()
            depend_dict[dep] = i
        return depend_dict,num_depend


if __name__ == '__main__':
    train_sentence_path = 'data/processed/train/sdp.txt'
    val_sentence_path = 'data/processed/val/sdp.txt'
    vocab_path = 'data/processed/vocab.txt'
    build_vocab_to_file_txt(train_sentence_path, val_sentence_path, vocab_path)
    vocab_dict, num_words = load_vocab_to_dict(vocab_path)
    print('Number of vocabularies:{} '.format(num_words))

import spacy
import nltk
import re
import spacy
import networkx as nx
import logging
import os
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
logging.basicConfig(level=logging.INFO)


pattern_symbol = re.compile('^[!"#$%&\\\'()*+,-./:;<=>?@[\\]^_`{|}~]|[!"#$%&\\\'()*+,-./:;<=>?@[\\]^_`{|}~]$')
pattern_entity1_start = re.compile("<e1>")
pattern_entity2_start = re.compile("<e2>")
pattern_entity1_end = re.compile("</e1>")
pattern_entity2_end = re.compile("</e2>")
class2label = {"Cause-Effect(e1,e2)":0,"Cause-Effect(e2,e1)":1,\
               "Instrument-Agency(e1,e2)":2,"Instrument-Agency(e2,e1)":3,\
               "Product-Producer(e1,e2)":4,"Product-Producer(e2,e1)":5,\
               "Content-Container(e1,e2)":6,"Content-Container(e2,e1)":7,\
               "Entity-Origin(e1,e2)":8,"Entity-Origin(e2,e1)":9,\
               "Entity-Destination(e1,e2)":10,"Entity-Destination(e2,e1)":11,\
               "Component-Whole(e1,e2)":12,"Component-Whole(e2,e1)":13,\
               "Member-Collection(e1,e2)":14,"Member-Collection(e2,e1)":15,
               "Message-Topic(e1,e2)":16,"Message-Topic(e2,e1)":17,"Other":18}

error_sdp = 0


def get_shortest_path_one_token(sentence,token1,token2):
    """
    Get shortest dependency path between 2 tokens
    @param sentence: the processed sentence
    @param token1: source token
    @param token2: des token
    @return: list of tokens denotes the shortest dependency path between source token and des token
    """
    global error_sdp
    doc = nlp(sentence.lower())
    edges = []
    new_token1 = token1
    new_token2 = token2

    # assign index for source token

    for token in doc:
        if token.lower_ == token1.lower():
            new_token1 = new_token1+'-{}'.format(token.i)
            break

    # assign index for des token

    for token in doc:
        if token.lower_ == token2.lower():
            new_token2 = new_token2 + '-{}'.format(token.i)
            break
    # Generate dependency tree
    for token in doc:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_, token.i),
                          '{0}-{1}'.format(child.lower_, child.i)))
    graph = nx.Graph(edges)
    try:
        # Get shortest path from source token to des token
        sdp = nx.shortest_path(graph, source=new_token1.lower(), target=new_token2.lower())
        list_depend = []
        tmp_sdp = sdp
        # Get depend for sdp
        for token in doc:
            for child in token.children:
                tmp1 = "{0}-{1}".format(token.lower_,token.i)
                tmp2 = "{0}-{1}".format(child.lower_,child.i)
                if tmp1 in tmp_sdp and tmp2 in tmp_sdp:
                    # print(token, child, tmp1,tmp2, child.dep_, token.dep_)
                    list_depend.append(child.dep_)
        return sdp,list_depend
    except Exception as e:
        # if not exists shortest dependency path between 2 tokens then assign sdp equals [ source token, des token]
        print(e)
        print(sentence)
        print("error:")
        print(edges)
        error_sdp += 1
        sdp = [token1,token2]
        list_depend = []
        for token in doc:
            if token.lower_ in sdp:
                list_depend.append(token.dep_)
        return sdp,list_depend


def get_shortest_path(sentence):
    """
    Get shortest dependency path and depend from one sentence which contains two entities
    @param sentence: The input sentence
    @return: the shortest dependency path between 2 entities
    """
    sentence,tmp1,tmp2,entity1,entity2 = process_sentence(sentence)
    shortest_path = None
    depend = None
    shortest_length = 10000
    max_length = -10000
    for i in tmp1:
        for j in tmp2:
            sdp,dep = get_shortest_path_one_token(sentence, i, j)
            t = len(sdp)
            u = len(dep)
            if t < shortest_length and i!=j:
                shortest_length = t
                shortest_path = sdp
            if u > max_length and i!=j:
                depend = dep
                max_length = u
    entity1 = ' '.join([x for x in entity1])
    entity2 = ' '.join([x for x in entity2])
    shortest_path[0] = entity1
    shortest_path[-1] = entity2
    return shortest_path,depend


def process_sentence(sentence):
    """
    preprocess the sentence and extract entities from the sentence
    @param sentence: the input sentence
    @return: preprocessed sentence, entity1 with index, entity2 with index , entity1 , entity2
    """
    sentence = sentence.strip()
    # print('original: ', sentence)
    sentence = pattern_entity1_start.sub('e1 ',sentence)
    sentence = pattern_entity1_end.sub(' /e1',sentence)
    sentence = pattern_entity2_start.sub('e2 ',sentence)
    sentence = pattern_entity2_end.sub(' /e2',sentence)
    doc = nlp(sentence)
    list_token = list([x.lower_ for x in doc])
    try:
        e1_start = list_token.index('e1')
        e1_end = list_token.index('/e1')
        e2_start = list_token.index('e2')
        e2_end = list_token.index('/e2')
    except Exception as e:
        print(e)
        print(sentence)
    entity1 = list_token[e1_start+1:e1_end]
    entity2 = list_token[e2_start+1:e2_end]
    i = e1_start
    e1_final = []
    e2_final = []
    i = e1_start + 1
    while list_token[i] != '/e1':
        tmp = '{}-{}'.format(list_token[i],i-1)
        e1_final.append(tmp)
        i+=1
    i = e2_start + 1
    while list_token[i] != '/e2':
        tmp = '{}-{}'.format(list_token[i], i - 3)
        e2_final.append(tmp)
        i += 1

    delete = ['e1','/e1','e2','/e2',' ']
    final_sentence = []
    for token in doc:
        if token.lower_ not in delete:
            final_sentence.append(token.lower_)
    final_sentence = ' '.join([x for x in final_sentence])
    # print('final_sentence: ',final_sentence)
    return final_sentence,e1_final,e2_final,entity1,entity2


def clean_string_1(sentence):
    """
    preprocess the sentence
    @param sentence: the input sentence
    @return: the preprocessed sentence
    """
    sentence = sentence.lower()
    sentence = re.sub('<e',' <e',sentence)
    sentence = re.sub('/[0-9]/', '', sentence)
    sentence = re.sub('\.','',sentence)
    sentence = re.sub('km/h', 'kmh', sentence)
    sentence = re.sub(':', '', sentence)
    sentence = re.sub('-',' ',sentence)
    sentence = re.sub(',','',sentence)
    sentence = re.sub('\(','',sentence)
    sentence = re.sub('\)', '', sentence)
    sentence = re.sub('\'s','',sentence)
    sentence = re.sub('\'m', ' am', sentence)
    sentence = re.sub('don\'t', 'does not', sentence)
    sentence = re.sub('\'ve', ' have', sentence)
    sentence = re.sub(';', '', sentence)
    sentence = re.sub('%','',sentence)
    sentence = re.sub('    ', ' ', sentence)
    sentence = re.sub('   ', ' ', sentence)
    sentence = re.sub('  ', ' ', sentence)
    if pattern_symbol.search(sentence):
        return pattern_symbol.sub('',sentence)
    return sentence


def clean_string_2(sentence):
    """
    preprocess the sentence 2
    @param sentence: the input sentence
    @return: the preprocessed sentence
    """
    sentence = sentence.strip()
    sentence = re.sub('"', '', sentence)
    return sentence


def post_processed(sdp):
    """
    post processing the shortest dependency path
    @param sdp: list of token denotes the shortest dependency path between 2 entities
    @return: the post processed shortest dependency path
    """
    new_sdp = []
    for i,token in enumerate(sdp):
        if '-' not in token:
            new_sdp.append(token)
            continue
        else:
            idx = token.find("-")
            token = token[:idx]
            new_sdp.append(token)
    return new_sdp


def mapping_sdp_pos(pos,sdp):
    """
    mapping the part of speech tagging with the shortest dependency path
    @param pos: list of tokens denotes the part of speech tagging of the sentence
    @param sdp: list of tokens denotes the shortest dependency path
    @return: list of tokens denotes the part of speech tagging of the shortest dependency path
    """
    labels = []
    keys = [x[0] for x in pos]
    values = [x[1] for x in pos]
    dic = dict(zip(keys,values))
    new_sdp = sdp.copy()
    e1 = new_sdp[0]
    e2 = new_sdp[-1]
    e1 = e1.split(" ") if " " in e1 else [e1]
    e2 = e2.split(" ") if " " in e2 else [e2]
    e1.extend(new_sdp[1:-1])
    e1.extend(e2)
    for token in e1:
        if token in dic.keys():
                labels.append(dic[token])
    return labels


def split_to_sentences_and_labels(file_path,sentences_path,labels_path):
    """
    Split the raw data to sentences file and labels file
    @param file_path: the path to the raw data file
    @param sentences_path: the sentences file path
    @param labels_path: the label file path
    @return: None
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError('Your File Path Is Not Found')
    with open(file_path,'r') as f:
        lines = f.readlines()
        sentences = []
        labels = []
        count = 0
        while count + 4 < len(lines):
            sentence = lines[count].split("\t")[1]
            label = lines[count+1]
            sentence = clean_string_1(sentence)
            sentences.append(sentence)
            labels.append(label)
            count = count + 4
    assert len(sentences) == len(labels)
    logging.info('Writing sentences to file......')
    with open(sentences_path,'w') as f:
        for sent in sentences:
            f.writelines(sent)
    logging.info('Writing labels to file........')
    with open(labels_path,'w') as f:
        for label in labels:
            f.writelines(label)
    logging.info('Done !')


def generate_pos_tag_for_sentence(sentence):
    """
    Generate part of speech tagging for the input sentence
    @param sentence: the input sentence
    @return: list of tokens denotes the part of speech tagging of the input sentence
    """
    sentence,tmp1,tmp2,entity1,entity2 = process_sentence(sentence)
    words_list = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words_list)
    return tagged


def generate_pos_tag_for_sdp(sentence,sdp):
    """
    Generate part speech tagging for shortest dependency path
    @param sentence: the input sentence
    @param sdp: the shortest dependency path between 2 entities
    @return: list of tokens denotes the part of speech tagging for shortest dependency path
    """
    pos_tag = generate_pos_tag_for_sentence(sentence)
    sdp_pos_tag = mapping_sdp_pos(pos_tag,sdp)
    return sdp_pos_tag


def convert_sentence_to_sdp(sentence_path,sdp_path,sdp_pos_tag_path,depend_path):
    """
    Generate the sdp,pos,depend from raw data
    @param sentence_path: the sentences file path
    @param sdp_path: the shortest dependency file path
    @param sdp_pos_tag_path: the shortest dependency pos file path
    @param depend_path: the shortest dependency depend file path
    @return: None
    """
    logging.info('Converting sentences to sdp......')
    max_length_sdp = -10000
    with open(sentence_path,'r') as f:
        sentences = f.readlines()
        list_sdp = []
        list_depend = []
        list_sdp_pos_tag = []
        for sentence in sentences:
            sdp,depend = get_shortest_path(sentence)
            sdp = post_processed(sdp)
            if len(sdp) > max_length_sdp:
                max_length_sdp = len(sdp)
            sdp_pos_tag = generate_pos_tag_for_sdp(sentence,sdp)
            # assert len(sdp) == len(sdp_pos_tag)
            sdp = ' '.join([x for x in sdp])
            depend = ' '.join([x for x in depend])
            sdp_pos_tag = ' '.join([x for x in sdp_pos_tag])
            list_sdp.append(sdp)
            list_depend.append(depend)
            list_sdp_pos_tag.append(sdp_pos_tag)
    logging.info('Writing sdp to file......')
    assert len(list_sdp) == len(list_sdp_pos_tag)
    assert len(list_sdp) == len(list_depend)
    with open(sdp_path,'w') as f:
        for sdp in list_sdp:
            f.write(sdp+'\n')
    with open(sdp_pos_tag_path,'w') as f:
        for sdp_pos in list_sdp_pos_tag:
            f.write(sdp_pos+'\n')
    with open(depend_path,'w') as f:
        for dep in list_depend:
            f.write(dep+'\n')
    logging.info("Max length sdp:{}".format(max_length_sdp))
    logging.info('Done !')


def generate_processed_data(file_path, sentences_path,\
                            labels_path,sdp_path,sdp_pos_path,depend_path):
    """
    Generate processed data
    @param file_path: raw file path
    @param sentences_path: sentence file path
    @param labels_path: label file path
    @param sdp_path: sdp file path
    @param sdp_pos_path: sdp pos file path
    @param depend_path: sdp depend file path
    @return: None
    """
    split_to_sentences_and_labels(file_path, sentences_path, labels_path)
    convert_sentence_to_sdp(sentences_path,sdp_path,\
                            sdp_pos_path,depend_path)
    logging.info('Error From Extracting Shortest Dependency Path: {}'.format(error_sdp))


def main():
    # Generate train data
    logging.info('Generate train data .....')
    train_file_path = 'data/raw/SemEval2010_task8_training/TRAIN_FILE.TXT'
    train_sentences_path = 'data/processed/train/sentences.txt'
    train_labels_path = 'data/processed/train/labels.txt'
    train_sdp_path = 'data/processed/train/sdp.txt'
    train_sdp_pos_path = 'data/processed/train/sdp_pos.txt'
    train_depend_path = 'data/processed/train/depend.txt'
    generate_processed_data(train_file_path, train_sentences_path, train_labels_path, train_sdp_path, \
                            train_sdp_pos_path, train_depend_path)

    # Train SDP Error: 23
    # Generate validation data
    logging.info('Generate validation data .....')
    val_file_path = 'data/raw/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    val_sentences_path = 'data/processed/val/sentences.txt'
    val_labels_path = 'data/processed/val/labels.txt'
    val_sdp_path = 'data/processed/val/sdp.txt'
    val_sdp_pos_path = 'data/processed/val/sdp_pos.txt'
    val_depend_path = 'data/processed/val/depend.txt'

    generate_processed_data(val_file_path, val_sentences_path, val_labels_path, val_sdp_path, \
                            val_sdp_pos_path, val_depend_path)
    # Val SDP Error:16
    logging.info('Error From Extracting Shortest Dependency Path: {}'.format(error_sdp))

    # Total SDP Error: 39


if __name__ == '__main__':
    main()

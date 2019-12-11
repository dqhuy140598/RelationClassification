## Relation Classification 

### Installation

**1. My Code:**

- Frist open google colab with GPU

- Clone this project to Google Colab:

    ``!git clone https://github.com/dqhuy140598/RelationClassification.git``

- Download pretrained Word2Vec:

    ``!wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz``

- Unzip pretrained Word2Vec:

    ``!gunzip GoogleNews-vectors-negative300.bin.gz``

- Cd to this project:

    ``%cd RelationClassification/``
    
- Generate training and validation shortest dependency path and part of speech tagging:

    ``!python ./utils/data_utils.py``
    
    (data/processed/)

- Build words vocabulary:

    ``!python ./utils/build_vocab.py``
    
    (data/processed/vocab.txt)

- Train model:
    
    ``!python train.py --pretrained --pretrained $PRETRAINED_PATH$ --cal_thresh True``
    
 

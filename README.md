## Relation Classification 

### Installation

**1. My Code:**

- Frist open google colab with GPU

- Clone this project to Google Colab:

    ```bash
    !git clone https://github.com/dqhuy140598/RelationClassification.git
    ```

- Download pretrained Word2Vec:

    ```bash
    !wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
    ```

- Unzip pretrained Word2Vec:

    ```bash
    !gunzip GoogleNews-vectors-negative300.bin.gz
    ```

- Cd to this project:

    ```bash
    %cd RelationClassification/
    ```
    
- Generate training and validation shortest dependency path and part of speech tagging:

    ```bash
    !python ./utils/data_utils.py
    ```
    
    (data/processed/)

- Build words vocabulary:

    ```bash
    !python ./utils/build_vocab.py
    ```
    
    (data/processed/vocab.txt)

- Train model:

    - If you use threshold equal to 0.5:
    
        ```bash
        !python train.py --pretrained $PRETRAINED_PATH$
        ```
    
    - Else:
        
        ```bash
        !python train.py --pretrained $PRETRAINED_PATH$ --use_thresh True
        ```

    -   $PRETRAINED_PATH$ : PATH to pretrained Word2Vec. Please replace it by **/content/GoogleNews-vectors-negative300.bin**
    
**2. EMNLP2017 Code:**

- Convert the processed data to use for this code:

    ```bash
    !python ./utils/convert.py
    ```

- Train model:

    - If you use threshold equal to 0.5:
    
        ```bash
        !python train.py --data $DATA_PATH$ --pretrained $PRETRAINED_PATH$
        ```
    
    - Else:
        
        ```bash
        !python train.py --data $DATA_PATH$ --pretrained $PRETRAINED_PATH$ --use_thresh True
        ```

    -   $PRETRAINED_PATH$ : PATH to the pretrained Word2Vec. Please replace it by **/content/GoogleNews-vectors-negative300.bin**
    -   $DATA_PATH$: PATH to the converted data. please replace it by **data_convert.json**





 

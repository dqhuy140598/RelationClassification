import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self, word_embeddings, pos_size, depend_size, params):
        """
        constructor of CNN model class
        @param word_embeddings: embedding matrix of words vocabulary
        @param pos_size: part of speech tagging size
        @param depend_size: depend size
        @param params: hyper parameters of the cnn model
        """
        super(CNN, self).__init__()

        self.word_embedding = torch.nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(word_embeddings),
                                                                 freeze=False)
        self.pos_embedding = torch.nn.Embedding(num_embeddings=pos_size, embedding_dim=params['pos_embedding_size'])
        # self.depend_embedding = torch.nn.Embedding(num_embeddings=depend_size,embedding_dim=params['depend_embedding_size'])

        drop_out_rate = params['drop_out']

        self.drop_out = torch.nn.Dropout(p=drop_out_rate)

        self.convs = []

        # feature_dims = params['embedding_size'] + params['pos_embedding_size'] + params['depend_embedding_size']

        feature_dims = params['embedding_size'] + params['pos_embedding_size']

        for filter_size in params['filters_size']:
            conv = torch.nn.Conv2d(in_channels=1, out_channels=params['n_filters'],
                                   kernel_size=(filter_size, feature_dims))

            self.convs.append(conv)

        flat_size = params['n_filters'] * len(params['filters_size'])

        n_classes = params['n_classes']

        self.linear0 = torch.nn.Linear(flat_size, 250)

        self.linear = torch.nn.Linear(250, n_classes)

        # Binary cross entropy loss for binary classification problem
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        """
        feed forward the input
        @param x: input data
        @return: output data
        """

        # use only shortest dependency path and part of speech tagging
        sdp_idx = x[0]
        pos_sdp_idx = x[1]
        # depend_idx = x[2]
        # position1 = x[2]
        # position2 = x[3]

        sdp_embeddings = self.word_embedding(sdp_idx)
        pos_sdp_embeddings = self.pos_embedding(pos_sdp_idx)
        # depend_embeddings = self.depend_embedding(depend_idx)
        # pos1_embeddings = self.position_embedding_1(position1)
        # pos2_embeddings = self.position_embedding_2(position2)

        feature = torch.cat([sdp_embeddings, pos_sdp_embeddings], dim=2)

        # feature = sdp_embeddings

        feature = torch.unsqueeze(feature, dim=1)  # batch_size x 1 x max_length x (embedding_size + pos_size)

        out_feature = []

        for conv in self.convs:
            out = conv(feature)  # batch_size x n_filters x (max_length - filter_size + 1) x 1

            out = torch.squeeze(out, dim=-1)  # batch_size x n_filters x (max_length - filter_size + 1)

            out_feature.append(out)

        out_feature = [torch.relu(x) for x in out_feature]

        out_pool = [F.max_pool1d(x, kernel_size=(x.size(2))).squeeze(dim=2) for x in
                    out_feature]  # [batch_size x n_filters]

        sentence_features = torch.cat(out_pool, dim=1)

        output = self.drop_out(sentence_features)

        output = self.linear0(output)

        output = self.linear(output)

        return output
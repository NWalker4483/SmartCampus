import torch
import torch.nn as nn
from torchreid.utils import FeatureExtractor

class LeNetHead(nn.Module):
    def __init__(self, input_shape):
        self.sequence_len = input_shape[0]
        # call the parent constructor
        super(LeNetHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=16, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5)),
            nn.Flatten()
        )

        self.output_size = self.conv(
            torch.rand((1, *input_shape))).data.shape[1]

    def forward(self, x):
        return self.conv(x)

class OSNet(nn.Module):
    def __init__(self):
        
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )
    def forward(self):
        pass

class CameraFeedHead(nn.Module):
    def __init__(self, seq_shape, handcraft_feat_len, embed_len=64):
        super(CameraFeedHead, self).__init__()
        # Defining some parameters

        self.seq_len = seq_shape[0]
        self.img_shape = seq_shape[1:]

        self.hidden_dim = 500

        self.cnn = LeNetHead(self.img_shape)

        # More Blah
        self.rnn = nn.RNN(self.cnn.output_size, 500, batch_first=False)

        # Temporal Pooling Layer
        # torch.mean(rnn_out)

        # Padded Fully connected layer
        self.fc = nn.Linear(self.hidden_dim + handcraft_feat_len, 128)

        # Second Fully connected layer
        self.fc2 = nn.Linear(128, embed_len)

    def forward(self, images, handcrafted_features):
        # Convert the list of sequences into a list of the ith image in a sequence
        image_representations = []

        for image in torch.transpose(images, 0, 1):
            image_representations.append(self.cnn(image))

        # Convert Back to Tensor
        image_representations = torch.stack(image_representations)

        # Passing in the input and hidden state into the model and obtaining outputs
        temporal_representations, _ = self.rnn(image_representations)

        # Temporal Pooling
        sequence_representation = torch.mean(temporal_representations, 0)

        # Average all image representations across time
        joined_representation = torch.cat((sequence_representation, handcrafted_features), dim=1)

        out_ = self.fc(joined_representation)
        out = self.fc2(out_)

        return out


class EmbeddingComparisonModel(nn.Module):
    def __init__(self, embed_len):
        super(EmbeddingComparisonModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear((embed_len * 2) + 1, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embed_A, time_A, embed_B, time_B):
        x = torch.cat([embed_A, torch.abs(time_A - time_B), embed_B], dim=1)
        return self.model(x)


class SiameseModelWrapper(nn.Module):
    def __init__(self, seq_shape, handcraft_feat_len, embed_len):
        super(SiameseModelWrapper, self).__init__()
        
        self.query = CameraFeedHead(
            seq_shape, handcraft_feat_len, embed_len)
        self.compare = EmbeddingComparisonModel(embed_len)

    def forward(self, img1, feat1, img2, feat2):
        # Extract frame numbers into seperate tensor
        t1 = feat1[:, 0].reshape((-1, 1))
        t2 = feat2[:, 0].reshape((-1, 1))

        # Remove frame info from feature tensor
        feat1 = feat1[:, 1:]
        feat2 = feat2[:, 1:]

        q1 = self.query(img1, feat1)
        q2 = self.query(img2, feat2)
        out = self.compare(q1, t1, q2, t2)
        return q1, q2, out


class SiameseMultiHeadModelWrapper(nn.Module):
    def __init__(self, seq_shape, handcraft_feat_len, embed_len, heads=2):
        super(SiameseMultiHeadModelWrapper, self).__init__()
        self.heads = []
        for _ in range(heads):
            self.heads = CameraFeedHead(
                seq_shape, handcraft_feat_len, embed_len)
        self.compare = EmbeddingComparisonModel(embed_len)

    def forward(self, img1, feat1, img2, feat2):
        # Extract frame numbers into seperate tensor
        t1 = feat1[:, 0].reshape((-1, 1))
        t2 = feat2[:, 0].reshape((-1, 1))

        # Remove frame info from feature tensor
        feat1 = feat1[:, 1:]
        feat2 = feat2[:, 1:]
        q1 = self.query(img1, feat1)
        q2 = self.query(img2, feat2)
        out = self.compare(q1, t1, q2, t2)
        return out


if __name__ == "__main__":
    batch_size = 1
    input_shape = (5, 64, 64)

    rnd_imgs = torch.rand((1, *input_shape))

    test = LeNetHead(input_shape)
    print(test(rnd_imgs))

    seq_shape = (8, 5, 64, 64)
    handcraft_feat_len = 8 + 2
    embed_len = 64

    rnd_seqs = torch.rand((2, *seq_shape))
    rnd_feats = torch.rand((2, handcraft_feat_len))

    test = CameraFeedHead(seq_shape, handcraft_feat_len, embed_len)
    print(test(rnd_seqs, rnd_feats))

    embed_len = 64

    rnd_embed_A = torch.rand((batch_size, embed_len))
    rnd_time_A = torch.rand((batch_size, 1))

    rnd_embed_B = torch.rand((batch_size, embed_len))
    rnd_time_B = torch.rand((batch_size, 1))

    test = EmbeddingComparisonModel(embed_len)
    print(test(rnd_embed_A, rnd_time_A, rnd_embed_B, rnd_time_B))

    batch_size = 2
    embed_len = 64
    seq_shape = (8, 5, 128, 128)
    handcraft_feat_len = 8 

    rnd_seqs_A = torch.rand((batch_size, *seq_shape),requires_grad= True)
    rnd_feats_plus_time_A = torch.rand((batch_size, handcraft_feat_len + 1),requires_grad= True)

    rnd_seqs_B = torch.rand((batch_size, *seq_shape),requires_grad= True)
    rnd_feats_plus_time_B = torch.rand((batch_size, handcraft_feat_len + 1),requires_grad= True)

    test = SiameseEmbeddingModel(
        seq_shape=seq_shape, handcraft_feat_len=handcraft_feat_len, embed_len=embed_len)

    print(test(rnd_seqs_A, rnd_feats_plus_time_A,
          rnd_seqs_B, rnd_feats_plus_time_B))

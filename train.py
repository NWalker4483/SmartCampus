import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SequenceTripletDataset, SequencePairDataset
from model import SiameseEmbeddingModel
import time
import numpy as np
import gflags
import sys
from collections import deque
import os

if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", False, "use cuda")
    gflags.DEFINE_string("train_path", "raw_data", "training folder")
    gflags.DEFINE_string("test_path", "raw_data", 'path of testing folder')
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer(
        "show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer(
        "save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer(
        "test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_integer(
        "max_iter", 5000, "number of iterations before stopping")
    gflags.DEFINE_string(
        "model_path", "raw_data/models/sem", "path to store model")
    gflags.DEFINE_string("gpu_ids", "", "gpu ids used to train ex. 0,1,2,3")

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        # transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")

    train_set = SequenceTripletDataset(
        Flags.train_path)
    test_set = SequencePairDataset(
        Flags.test_path)

    test_loader = DataLoader(
        test_set, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    # TODO: Research custom sampler inorder to provide harder triplets 
    train_loader = DataLoader(
        train_set, batch_size=Flags.batch_size, shuffle=True, num_workers=Flags.workers)

    mean_squared_loss = torch.nn.MSELoss()
    triplet_margin_loss = torch.nn.TripletMarginLoss(margin=.1, swap=False)

    embed_len = 64
    seq_shape = (8, 5, 128, 128)
    handcraft_feat_len = 8 + 2

    net = SiameseEmbeddingModel(
        seq_shape=seq_shape, handcraft_feat_len=handcraft_feat_len, embed_len=embed_len)

    ones = torch.ones(Flags.batch_size, 1)
    zeros = torch.zeros(Flags.batch_size, 1)
    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=Flags.lr)
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (negative, anchor, positive) in enumerate(train_loader, 1):
        if batch_id > Flags.max_iter:
            break

        optimizer.zero_grad()

        negative_seq, negative_feat = negative
        anchor_seq, anchor_feat = anchor
        positive_seq, positive_feat = positive

        if Flags.cuda:
            negative_seq, negative_feat = negative_seq.cuda(), negative_feat.cuda()
            anchor_seq, anchor_feat = anchor_seq.cuda(), anchor_feat.cuda()
            positive_seq, positive_feat = positive_seq.cuda(), positive_feat.cuda()

        anchor_embed, positive_embed, scores = net.forward(
            anchor_seq, anchor_feat, positive_seq, positive_feat)

        loss = mean_squared_loss(scores, ones)

        anchor_embed, negative_embed, scores = net.forward(
            anchor_seq, anchor_feat, positive_seq, positive_feat)

        loss += mean_squared_loss(scores, zeros)

        loss += triplet_margin_loss(positive_embed,
                                    anchor_embed, negative_embed)

        loss_val += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()

        if batch_id % Flags.show_every == 0 or (batch_id <= 5):
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                batch_id, loss_val/Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path +
                       '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % Flags.test_every == 0:
            right, error = 0, 0
            # for _, ((seqA, featA), (seqB, featB)) in enumerate(test_loader, 1):
            #     if Flags.cuda:
            #         imgA, imgB = imgA.cuda(), imgB.cuda()
            #     output = net.forward(imgA, imgB).data.cpu().numpy()
            #     pred = np.argmax(output)
            #     if pred == 0:
            #         right += 1
            #     else:
            #         error += 1
            print('*' * 70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (
                batch_id, right, error, right*1.0/(right+error)))
            print('*' * 70)
            queue.append(right*1.0/(right+error))
        train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)

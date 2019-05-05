from data.mfcc_dataloader import MFCC_dataset
from torch.utils.data import DataLoader
from models.models import X_vector
import argparse
from torch import nn
import torch
def main(opt):
    x_vector = X_vector()
    mfcc_dataset = MFCC_dataset('data/data')
    mfcc_dataloader = DataLoader(dataset=mfcc_dataset, batch_size=opt['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(x_vector.parameters(), lr=0.001)
    entropy_loss = nn.CrossEntropyLoss()
    for epoch in range(opt['epoches']):
        for mfcc,gender in mfcc_dataloader:
            gender = gender.squeeze(1)
            gender_predict = x_vector(mfcc)
            # print(gender.shape)
            loss = entropy_loss(gender_predict, gender)
            loss.backward()
            optimizer.step()
            print("epoch %d: loss is %.3f" % (epoch, loss.item()))
        # calculate accuracy
        acc = 0
        for mfcc,gender in mfcc_dataloader:
            gender_predict = x_vector(mfcc)
            acc+=(torch.max(gender_predict,1)[1]==gender).float().mean().item()

        accuracy=acc / float(len(mfcc_dataset))
        print("epoch %d: accuracy is %.3f"%(epoch,accuracy))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoches', type=int, default=4)
    opt = vars(parser.parse_args())
    main(opt)
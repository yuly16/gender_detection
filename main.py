from data.mfcc_dataloader import MFCC_dataset
from torch.utils.data import DataLoader
from models.models import X_vector
import argparse
from torch import nn
from tqdm import tqdm
import torch
def main(opt):
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:1') if use_cuda else torch.device('cpu')
    x_vector = X_vector()
    x_vector = x_vector.to(device)
    mfcc_dataset = MFCC_dataset('/mnt/workspace2/yuly/gender_data/train')
    mfcc_test_dataset = MFCC_dataset('/mnt/workspace2/yuly/gender_data/test')
    mfcc_dataloader = DataLoader(dataset=mfcc_dataset, batch_size=opt['batch_size'], shuffle=True)
    mfcc_test_dataloader = DataLoader(dataset=mfcc_test_dataset, batch_size=opt['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(x_vector.parameters(), lr=2e-5)
    entropy_loss = nn.CrossEntropyLoss()
    for epoch in range(opt['epoches']):
        print('training process:')
        x_vector.train()
        for mfcc,gender in tqdm(mfcc_dataloader):
            #print(mfcc.shape)
            mfcc = mfcc.to(device)
            gender = gender.to(device)
            gender = gender.squeeze(1)
            gender_predict = x_vector(mfcc)
            loss = entropy_loss(gender_predict, gender)
            loss.backward()
            optimizer.step()
        print("epoch %d: loss is %.3f" % (epoch, loss.item()))
        # calculate accuracy
        acc = 0
        x_vector.eval()
        print('testing process:')
        for mfcc,gender in tqdm(mfcc_test_dataloader):
            mfcc = mfcc.to(device)
            gender = gender.to(device)
            gender = gender.squeeze(1)
            gender_predict = x_vector(mfcc)
            acc += (torch.max(gender_predict,1)[1]==gender).float().sum().item()
        accuracy=acc / float(len(mfcc_test_dataset))
        print("epoch %d: accuracy is %.3f"%(epoch,accuracy))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoches', type=int, default=40)
    opt = vars(parser.parse_args())
    main(opt)

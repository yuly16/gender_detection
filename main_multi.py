from data.mfcc_dataloader import MFCC_dataset
from torch.utils.data import DataLoader
from models.models import X_vector
import argparse
from torch import nn
from tqdm import tqdm
import random
import torch
import os
from utils.util import *
def main(opt):
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    x_vector = X_vector()
    x_vector = x_vector.to(device)
    mfcc_dataloaders = mk_dataloaders(opt,'/mnt/workspace2/yuly/gender_data/train_ch')
    mfcc_test_dataset = MFCC_dataset('/mnt/workspace2/yuly/gender_data/test')
    mfcc_test_dataloader = DataLoader(dataset=mfcc_test_dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(x_vector.parameters(), lr=1e-5)
    entropy_loss = nn.CrossEntropyLoss()
    for epoch in range(opt['epoches']):
        print('training procedure:')
        for n_job,mfcc_dataloader in mfcc_dataloaders:
            print("selecting dataloader...")
            x_vector.train()
            for mfcc,gender in tqdm(mfcc_dataloader):
                #print(mfcc.shape)
                mfcc = mfcc.to(device)
                gender = gender.to(device)
                gender = gender.squeeze(1)
                gender_predict = x_vector(mfcc)
                loss = entropy_loss(gender_predict, gender)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(x_vector.parameters(),5)
                optimizer.step()
            acc = 0
            loss = 0
            x_vector.eval()
        print('testing procedure:')
        with torch.no_grad():
            for mfcc,gender in tqdm(mfcc_test_dataloader):
                mfcc = mfcc.to(device)
                gender = gender.to(device)
                gender = gender.squeeze(1)
                gender_predict = x_vector(mfcc)
                loss += entropy_loss(gender_predict, gender)
                acc += (torch.max(gender_predict,1)[1]==gender).float().sum().item()
            accuracy=acc / float(len(mfcc_test_dataset))
            print("epoch %d, n_job %d: accuracy is %.3f; loss is %.3f"%(epoch,n_job,accuracy,loss))
            save_model(epoch,n_job,accuracy,x_vector)
            update_log(epoch,n_job,accuracy,loss)


        # print("epoch %d: loss is %.3f" % (epoch, loss.item()))
        # # calculate accuracy
        # acc = 0
        # loss = 0
        # x_vector.eval()
        # print('testing process:')
        # for mfcc,gender in tqdm(mfcc_test_dataloader):
        #     mfcc = mfcc.to(device)
        #     gender = gender.to(device)
        #     gender = gender.squeeze(1)
        #     gender_predict = x_vector(mfcc)
        #     loss += entropy_loss(gender_predict, gender)
        #     acc += (torch.max(gender_predict,1)[1]==gender).float().sum().item()
        # accuracy=acc / float(len(mfcc_test_dataset))
        # print("epoch %d: accuracy is %.3f"%(epoch,accuracy))
        # print("epoch %d: loss is %.3f"%(epoch,loss))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoches', type=int, default=40)

    opt = vars(parser.parse_args())
    main(opt)

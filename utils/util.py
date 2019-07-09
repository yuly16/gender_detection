import os
import random
from data.mfcc_dataloader import MFCC_dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
def mk_dataloaders(opt,path):
    n_dataloaders = len(os.listdir(path))
    mfcc_dataloaders = []
    loader_ids = [i for i in range(n_dataloaders)]
    random.shuffle(loader_ids)
    for i in range(n_dataloaders):
        mfcc_dataset = MFCC_dataset(os.path.join(path,os.listdir(path)[loader_ids[i]]))
        mfcc_dataloaders.append((i,DataLoader(dataset=mfcc_dataset, batch_size=opt['batch_size'], shuffle=True)))
    return mfcc_dataloaders

def init_log():
    #如果log存在就删掉
    if os.path.exists('log/log.txt'):
        os.system('rm log/log.txt')
def update_log(epoch,n_job,acc,loss):
    log_path = os.path.join(os.getcwd(),'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    with open(os.path.join(log_path,'log.txt'),"a") as f:
        f.write("epoch: %d n_job: %d acc: %.3f loss: %.3f\n"%(epoch,n_job,acc,loss))
        f.close()
def save_model(epoch,n_job,acc,model):
    log_path = os.path.join(os.getcwd(),'log')
    if os.path.exists(os.path.join(log_path,'log.txt')):
        with open(os.path.join(log_path,'log.txt'),"r") as f:
            results = f.readlines()
            best_acc = np.max([float(result.split(' ')[5]) for result in results])
            if acc > best_acc:
                torch.save(model,'log/best_model.pth')
                print("best model, saving...")
    else:
        torch.save(model,'log/best_model.pth')
        print("best model, saving...")
def load_model():
    pass


  
    

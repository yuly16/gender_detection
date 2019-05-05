from data.mfcc_dataloader import MFCC_dataset
from torch.utils.data import DataLoader
import argparse
def main(opt):
    mfcc_dataset = MFCC_dataset('data/data')
    mfcc_dataloader = DataLoader(dataset=mfcc_dataset, batch_size=opt['batch_size'], shuffle=True)
    for mfcc,gender in mfcc_dataloader:







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    opt = vars(parser.parse_args())
    main(opt)
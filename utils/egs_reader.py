import argparse
import struct
import numpy as np
import os


u2g_dict = {}
def utt2gender(utt2spk_path,spk2gender_path):
    u2s_dict = {}
    s2g_dict = {}
    #utt2spk
    f = open(utt2spk_path,'r')
    u2s_list = f.readlines()
    f.close()
    for u2s_item in u2s_list:
        u2s_item = u2s_item.strip() #remove \n
        u2s_dict[u2s_item.split(' ')[0]] = u2s_item.split(' ')[1]
    # spk2gender
    f = open(spk2gender_path,'r')
    s2g_list = f.readlines()
    f.close()
    for s2g_item in s2g_list:
        s2g_item = s2g_item.strip() #remove \n
        s2g_dict[s2g_item.split(' ')[0]] = s2g_item.split(' ')[1]
    for i in u2s_dict:
        u2g_dict[i] = s2g_dict[u2s_dict[i]]

#
def ReadBasicType(fb):
    size = ord(struct.unpack('c',fb.read(1))[0])
    num = struct.unpack('i',fb.read(size))[0]
    return size, num

#
def ReadIdxVec(fb):
    _ , size_idx = ReadBasicType(fb)
    #idx(n,t,x)
    idx=[]
    for i in range(size_idx):
        if i==0:
            c=ord(struct.unpack('c',fb.read(1))[0])
            if c<125:
                idx.append((0,c,0))
            else:
                _,n=ReadBasicType(fb)
                _,t=ReadBasicType(fb)
                _,x=ReadBasicType(fb)
                idx.append((n,t,x))   
        else:
            c=ord(struct.unpack('c',fb.read(1))[0])
            if c<125:
                n=idx[-1][0]
                t=idx[-1][1]+c
                x=idx[-1][2]
                idx.append((n,t,x)) 
            else:
                _,n=ReadBasicType(fb)
                _,t=ReadBasicType(fb)
                _,x=ReadBasicType(fb)
                idx.append((n,t,x)) 
    return idx


def ReadFeature(fb):
    #header
    global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')])
    per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])
    
    #Mapping for percentiles in col-headers, 
    def uint16_to_float(value, min, range):
        return np.float32(min + range * 1.52590218966964e-05 * value)
    
    # Mapping for matrix elements,
    def uint8_to_float_v2(vec, p0, p25, p75, p100):
        # Split the vector by masks,
        mask_0_64 = (vec <= 64);
        mask_193_255 = (vec > 192);
        mask_65_192 = (~(mask_0_64 | mask_193_255));    # Sanity check (useful but slow...),

        # assert(len(vec) == np.sum(np.hstack([mask_0_64,mask_65_192,mask_193_255])))
        # assert(len(vec) == np.sum(np.any([mask_0_64,mask_65_192,mask_193_255], axis=0)))
        # Build the float vector,
        ans = np.empty(len(vec), dtype='float32')
        ans[mask_0_64] = p0 + (p25 - p0) / 64. * vec[mask_0_64]
        ans[mask_65_192] = p25 + (p75 - p25) / 128. * (vec[mask_65_192] - 64)
        ans[mask_193_255] = p75 + (p100 - p75) / 63. * (vec[mask_193_255] - 192)
        return ans
    
    globmin, globrange, rows, cols = np.frombuffer(fb.read(16), dtype=global_header, count=1)[0]
    
    
    col_headers = np.frombuffer(fb.read(cols*8), dtype=per_col_header, count=cols)
    data = np.reshape(np.frombuffer(fb.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows))
    
    mat = np.empty((cols,rows), dtype='float32')
    for i, col_header in enumerate(col_headers):
        col_header_flt = [ uint16_to_float(percentile, globmin, globrange) for percentile in col_header ]
        mat[i] = uint8_to_float_v2(data[i], *col_header_flt)
    
    #np.savetxt(_debug_dir+'/egs.txt',mat.T)
    
    #_info = fb.read(20)
    #print("head: %s"%(_info))
    return mat.T
    
    

def main():  
    parser = argparse.ArgumentParser(description='This is the script to convert kaldi egs into numpy array, for pytorch training')
    parser.add_argument('--input_dir',type=str,help='.scp and .ark dir')
    parser.add_argument('--output_dir', type=str, help='feat.npy and label.npy dir')
    parser.add_argument('--num',type=int,help = 'jobs')
    
    args = parser.parse_args()
    #######################################################################
    # initialize
    os.chdir('/mnt/workspace1/liyt/kaldi/egs/sre18/v2')
    save_path = args.output_dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path,'m'))
        os.mkdir(os.path.join(save_path,'f'))
    overall_path = '/mnt/workspace1/liyt/kaldi/egs/sre18/v2/data'
    data_names = ['fisher','sre','sre10','swbd']
    for data_name in data_names:
        data_path = os.path.join(overall_path,data_name)
        feat_path = os.path.join(data_path,'feats.scp')
        utt2spk_path = os.path.join(data_path,'utt2spk')
        spk2gender_path = os.path.join(data_path,'spk2gender')
        utt2gender(utt2spk_path,spk2gender_path)
    #######################################################################
    scp_dir=args.input_dir+'/egs.{}.scp'.format(args.num)
    ark_dir=args.input_dir+'/egs.{}.ark'.format(args.num)
    
    f_scp=open(scp_dir,'r')
    f_ark=open(ark_dir,'rb')

    label=[]
    mat=[]
    for line in f_scp:
        s=line.split()
        if len(s)!=2:
            continue
        #read label 
        label.append(['-'.join(s[0].split('-')[:-3])])
        # print('-'.join(s[0].split('-')[:-3])) 
        offset=int(s[1].split(':')[-1])
        #seek to the offset
        f_ark.seek(offset+2,0)
        #read '<Nnet3Eg> <NumIo> '
        _info = f_ark.read(18)
        #print("head: %s"%(_info))
        #read size and num
        size_NnetIo, num_NnetIo = ReadBasicType(f_ark)
        #read '<NnetIo> input <I1V> '
        _info = f_ark.read(21)
        #read Index vector
        idx=ReadIdxVec(f_ark)
        #read 'CM '
        _info=f_ark.read(3)
        mat.append(ReadFeature(f_ark))
    f_scp.close()
    f_ark.close()
    mat = np.array(mat)
    label = np.array(label)




    for i in range(len(label)):
        assert len(label) == len(mat)
        u_label = label[i][0]
        if u_label.split('-')[-1] == 'music' or \
                u_label.split('-')[-1] == 'noise' or \
                u_label.split('-')[-1] == 'reverb' or \
                u_label.split('-')[-1] == 'babble':
            u_label_origin = '-'.join(u_label.split('-')[:-1])
            if u_label_origin in u2g_dict:
                file_path = os.path.join(save_path,u2g_dict[u_label_origin],u_label + '.npy')
                np.save(file_path,mat[i])
        else:         
            if u_label in u2g_dict:
                file_path = os.path.join(save_path,u2g_dict[u_label],u_label + '.npy')
                np.save(file_path,mat[i])


if __name__=='__main__':
    main()

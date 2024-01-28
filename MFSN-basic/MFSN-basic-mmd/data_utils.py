import pandas as pd
import torch
import os
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader,TensorDataset,RandomSampler

class Input_example(object):
    def __init__(self,input_ids,mask_ids,label,seq_len):
        self.input_ids = input_ids
        self.mask_ids = mask_ids
        self.seq_len = seq_len
        self.label = label


def get_device(cuda_id):
    if torch.cuda.is_available():
        device = torch.device('cuda', cuda_id)
    else:
        device = torch.device('cpu')
    return device



def get_data(path):
    data_df = pd.read_csv(path)
    pairs_list = data_df["pairs"].to_list()
    labels_list = data_df["labels"].to_list()
    return pairs_list,labels_list



def get_feature(pairs_list,lables_list,max_seq,tokenizer):

    input_feature_list = []
    for indx,(pair,label) in enumerate(zip(pairs_list,lables_list)):
        if (indx+1)%200==0:
            print("write example %d of %d" %(indx+1,len(pairs_list)))
        if "[SEP]" in pair:
            pair = pair.replace("[SEP]","</s>")
        tokenizer_output = tokenizer(pair,max_length=max_seq,padding="max_length",truncation=True,return_tensors="pt")
        inputfeature = Input_example(input_ids=tokenizer_output["input_ids"][0].numpy(),
                                     mask_ids=tokenizer_output["attention_mask"][0].numpy(),
                                     seq_len = torch.count_nonzero(tokenizer_output["attention_mask"][0],dim=0).item(),
                                     label=label)
        input_feature_list.append(inputfeature)


    return input_feature_list


def distilbert_convert_examples_to_features(pairs, labels, max_seq_length, tokenizer, pad_token_id=0, cls_token='[CLS]',
                                      sep_token='[SEP]'):
    input_feature_list = []
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        if sep_token in pair:
            left = pair.split("[SEP]")[0]
            right = pair.split("[SEP]")[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more < len(rtokens):
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more < len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("bad example!")
                    continue
            tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]

        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        inputfeature = Input_example(input_ids=input_ids,
                                     mask_ids=input_mask,
                                     seq_len=sum(input_mask),
                                     label=label)
        input_feature_list.append(inputfeature)
    return input_feature_list








def get_data_loader(input_feature_list,batch_size,flag):
    #joint all input_ids
    all_input_ids = np.array([f.input_ids for f in input_feature_list])
    all_input_ids = torch.tensor(all_input_ids,dtype=torch.long)
    # joint all mask_ids
    all_mask_ids = np.array([f.mask_ids for f in input_feature_list])
    all_mask_ids = torch.tensor(all_mask_ids, dtype=torch.long)

    all_seq_len = torch.tensor([f.seq_len for f in input_feature_list], dtype=torch.long)
    all_label = torch.tensor([f.label for f in input_feature_list], dtype=torch.long)
    #TensorDataset is similar to zip
    dataset = TensorDataset(all_input_ids,all_mask_ids,all_seq_len,all_label)

    sampler = RandomSampler(dataset)

    if flag == "dev":
        # Read all data
        dataloader = DataLoader(dataset=dataset,sampler=sampler,batch_size=batch_size)
    else:
        dataloader = DataLoader(dataset=dataset,sampler=sampler,batch_size=batch_size,drop_last=True)
    return dataloader


def creat_root_path(name,seed):
    root_path = os.path.join("../result_basic_MMD",name,str(seed))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    else:
        print("exist!")
    return root_path

def save_loss(result,name,root_path):

    path = os.path.join(root_path,name+".csv")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(path):
        data_df = pd.DataFrame(result,index=[0])
        data_df.to_csv(path,index=False)
        print("save loss to ", path)
    else:
        df1 = pd.read_csv(path)
        data_df = pd.DataFrame(result, index=[0])
        data_df = pd.concat([df1,data_df],ignore_index=True)
        data_df.to_csv(path,index=False)
        print("save loss to ",path)


def save_prf(result,name,root_path):
    path = os.path.join(root_path,name+".csv")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(path):
        data_df = pd.DataFrame(result,index=[0])
        data_df.to_csv(path,index=False)
        print("save P,R,F1 to ", path)
    else:
        df1 = pd.read_csv(path)
        data_df = pd.DataFrame(result, index=[0])
        data_df = pd.concat([df1,data_df],ignore_index=True)
        data_df.to_csv(path, index=False)
        print("save P,R,F1 to ",path)



def save_Hyparam(parm_dic,root_path):
    result_path = os.path.join(root_path,"params.txt")
    with open(result_path, 'w') as f:
        for key, value in parm_dic.items():
            f.write(key)
            f.write(': ')
            f.write(str(value))
            f.write('\n')





def save_model(net,name):
    """Save trained model."""
    folder = os.path.join("../checkpoints")
    path = os.path.join(folder, name + "_parameter.pkl")
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(),path)
    print("save pretrained model to: {}".format(path))




def list_mean(lis:list):

    return sum(lis)/len(lis)





import sys
sys.path.append("../")
import data_utils
import os
from sklearn.model_selection import train_test_split
import argparse
from transformers import DistilBertTokenizer
from models.DSN import DSN
from trainer.train import train
from draw_figure.get_loss import get_loss
from draw_figure.get_tgt_prf import get_tgt_prf




def do_ER(src_dataset,tgt_dataset,batch_size,BERT_type,seed,args):

    root_path=data_utils.creat_root_path(src_dataset+"-"+tgt_dataset,str(seed))

    #source domain dataset
    src_dataset_path = os.path.join("../data",src_dataset,src_dataset+".csv")
    src_pair_list,src_labels_lsit = data_utils.get_data(path=src_dataset_path)
    # target domain dataset
    tgt_dataset_path = os.path.join("../data",tgt_dataset,tgt_dataset+".csv")
    tgt_pair_list,tgt_labels_list = data_utils.get_data(path=tgt_dataset_path)
    # split datset: trainset:src_dataset; train_dataset:tgt_dataset(90%) test_dataset:tgt_dataset(10%)

    tgt_train_pairs_list,tgt_valid_pairs_list,tgt_train_labels_list,tgt_valid_labels_list = train_test_split(tgt_pair_list,
                                                                                                           tgt_labels_list,
                                                                                                           test_size=0.1,
                                                                                                           random_state=seed,
                                                                                                           stratify=tgt_labels_list)
    # in DADER,BERT='bert-base-multilingual-cased'
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_type)
    src_input_feature_list = data_utils.distilbert_convert_examples_to_features(pairs=src_pair_list,
                                                                            labels=src_labels_lsit,
                                                                            max_seq_length=args.max_seq,
                                                                            tokenizer=tokenizer)

    tgt_train_input_feature_list = data_utils.distilbert_convert_examples_to_features(pairs=tgt_train_pairs_list,
                                                     labels=tgt_train_labels_list,
                                                     max_seq_length=args.max_seq,
                                                     tokenizer=tokenizer)
    tgt_valid_input_feature_list = data_utils.distilbert_convert_examples_to_features(pairs=tgt_valid_pairs_list,
                                                     labels=tgt_valid_labels_list,
                                                     max_seq_length=args.max_seq,
                                                     tokenizer=tokenizer)


    #get data loader
    src_data_loader = data_utils.get_data_loader(input_feature_list=src_input_feature_list,
                                                 batch_size=batch_size,
                                                 flag="train")
    tgt_train_data_loader = data_utils.get_data_loader(input_feature_list=tgt_train_input_feature_list,
                                                       batch_size=batch_size,
                                                       flag="train")

    tgt_valid_data_loader = data_utils.get_data_loader(input_feature_list=tgt_valid_input_feature_list,
                                                      batch_size=batch_size,
                                                      flag="dev")

    #get model
    dsn = DSN(BERT_type)

    #train
    train(dsn=dsn,
          src_data_loader=src_data_loader,
          tgt_train_data_loader=tgt_train_data_loader,
          tgt_valid_data_loader=tgt_valid_data_loader,
          batch_size=batch_size,root_path=root_path,args=args)

    parm_dic = {"src":src_dataset,"tgt":tgt_dataset,"BERT": BERT_type, "batch_size": batch_size, "num_epochs": args.num_epochs, "max_seq": args.max_seq,
                "dann_epoch": args.dann_epoch, "lr": args.lr, "sim_weight":args.sim_weight, "dif_weight": args.dif_weight,
                "rec_weight": args.rec_weight,"seed":seed}
    data_utils.save_Hyparam(parm_dic,root_path=root_path)
    get_tgt_prf(dataset=src_dataset+"-"+tgt_dataset,root_path=root_path)
    get_loss(dataset=src_dataset+"-"+tgt_dataset,root_path=root_path)


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="b2",help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="fz",help="Specify tgt dataset")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")
    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")
    parser.add_argument('--num_epochs', type=int, default=60,
                        help="Specify random state")
    parser.add_argument('--dann_epoch', type=int, default=10,
                        help="Specify random state")
    parser.add_argument('--cuda_id', type=int, default=0,
                        help="Specify random state")

    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Specify random state")
    parser.add_argument('--sim_weight', type=float, default=0.01,
                        help="Specify random state")
    parser.add_argument('--dif_weight', type=float, default=1.0,
                        help="Specify random state")
    parser.add_argument('--rec_weight', type=float, default=0.05,
                        help="Specify random state")

    parser.add_argument('--max_seq', type=int, default=128,
                        help="Specify random state")
    parser.add_argument('--log_step', type=int, default=20,
                        help="Specify random state")

    return parser.parse_args()



def do_att_GRL(src,tgt,batch_size,seed,cuda_id):
    args = parse_arguments()
    BERT_type = "distilbert-base-uncased"
    args.cuda_id = cuda_id
    args.batch_size=batch_size
    args.src=src
    args.tgt=tgt
    args.seed = seed
    do_ER(src_dataset=src, tgt_dataset=tgt, BERT_type=BERT_type, batch_size=batch_size,seed=seed,args=args)




if __name__ == '__main__':
    args = parse_arguments()
    BERT_type = "distilbert-base-uncased"
    do_ER(src_dataset=args.src,tgt_dataset=args.tgt,BERT_type=BERT_type,batch_size=args.batch_size,seed=args.seed,args=args)
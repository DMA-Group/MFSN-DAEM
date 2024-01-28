import sys
sys.path.append("../")
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_utils import save_model,list_mean,save_prf,save_loss,get_device
import itertools
from functions import cal_coral_loss,DiffLoss
from trainer.evaluate import evaluate


def train(dsn,src_data_loader,tgt_train_data_loader,tgt_valid_data_loader,root_path,args):
    print("training!!!!!")
    bestf1 = 0
    # set DSN to "train" model
    dsn.train()
    #loss_function
    ERloss = nn.CrossEntropyLoss()
    Recloss = nn.CrossEntropyLoss()
    Difloss = DiffLoss()

    device = get_device(args.cuda_id)

    dsn = dsn.to(device)
    ERloss = ERloss.to(device)
    Recloss = Recloss.to(device)
    Difloss = Difloss.to(device)

    #define optimizer
    optimizer = optim.AdamW(dsn.parameters(), lr=args.lr)
    len_data_loader = min(len(src_data_loader), len(tgt_train_data_loader))


    for epoch in range(args.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_train_data_loader))
        #to print loss
        sim_loss_list = []
        src_dif_loss_list = []
        src_rec_loss_list = []
        src_er_loss_list = []
        tgt_dif_loss_list = []
        tgt_rec_loss_list = []
        totall_loss_list = []

        for step, (src, tgt) in data_zip:
            src_input_ids,src_mask_ids,src_seq_len,src_label = src
            tgt_input_ids, tgt_mask_ids,tgt_seq_len, _ = tgt

            src_input_ids = src_input_ids.to(device)
            src_mask_ids = src_mask_ids.to(device)
            src_label = src_label.to(device)

            tgt_input_ids = tgt_input_ids.to(device)
            tgt_mask_ids = tgt_mask_ids.to(device)

            optimizer.zero_grad()
            tgt_output = dsn(input_ids=tgt_input_ids,attention_mask=tgt_mask_ids,mode="tgt")
            src_output = dsn(input_ids=src_input_ids, attention_mask=src_mask_ids, mode="src")

            #different loss
            #tgt
            tgt_dif_loss = args.dif_weight * Difloss(tgt_output["private_feature"], tgt_output["share_feature"])
            #src
            src_dif_loss = args.dif_weight * Difloss(src_output["private_feature"], src_output["share_feature"])

            #rec loss
            #tgt
            tgt_lm_lables = tgt_input_ids[:,1:].contiguous()
            tgt_rec_loss = args.rec_weight*Recloss(tgt_output["lm_logits"].view(-1,dsn.config.vocab_size),tgt_lm_lables.view(-1))
            #src
            src_lm_lables = src_input_ids[:,1:].contiguous()
            src_rec_loss = args.rec_weight*Recloss(src_output["lm_logits"].view(-1,dsn.config.vocab_size),src_lm_lables.view(-1))

            #ER_loss
            src_er_loss = ERloss(src_output["ER_logits"], src_label)
            if epoch >= args.k_order_epoch:
                sim_loss = args.sim_weight*(cal_coral_loss(src_output["share_feature"],tgt_output["share_feature"]))
                loss = sim_loss+src_dif_loss+tgt_dif_loss+src_rec_loss+tgt_rec_loss+src_er_loss
            else:
                sim_loss=torch.tensor([0])
                loss = src_dif_loss+tgt_dif_loss+src_rec_loss+tgt_rec_loss+src_er_loss

            totall_loss_list.append(loss.item())
            sim_loss_list.append(sim_loss.item())
            src_er_loss_list.append(src_er_loss.item())
            src_dif_loss_list.append(src_dif_loss.item())
            src_rec_loss_list.append(src_rec_loss.item())
            tgt_dif_loss_list.append(tgt_dif_loss.item())
            tgt_rec_loss_list.append(tgt_rec_loss.item())

            loss.backward()
            optimizer.step()
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "Total_loss=%.4f ER_loss=%.4f Sim_loss=%.4f Diff_loss=%.6f Rec_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         loss.item(),
                         src_er_loss.item(),
                         sim_loss.item(),
                         src_dif_loss.item()+tgt_dif_loss.item(),
                         src_rec_loss.item()+tgt_rec_loss.item()
                         ))




        loss_result = {"epoch":epoch,"totall_loss":list_mean(totall_loss_list),"ER_loss":list_mean(src_er_loss_list),
                       "sim_loss":list_mean(sim_loss_list),
                       "src_dif_loss":list_mean(src_dif_loss_list),"tgt_dif_loss":list_mean(tgt_dif_loss_list),
                       "src_rec_loss":list_mean(src_rec_loss_list),"tgt_rec_loss":list_mean(tgt_rec_loss_list)}
        save_loss(result=loss_result,name="loss",root_path=root_path)

        # src set
        src_R, src_P, src_F = evaluate(model=dsn, data_loader=src_data_loader,device=device)
        # vaild set
        valid_R, valid_P, valid_F = evaluate(model=dsn, data_loader=tgt_valid_data_loader,device=device)
        # test set
        test_R, test_P, test_F = evaluate(model=dsn, data_loader=tgt_train_data_loader,device=device)
        if valid_F>bestf1:
            save_model(net=dsn,name=args.src+"-"+args.tgt+"-basic-k_order")
            bestf1 = valid_F

        src_result = {"epoch":epoch,"src_R":src_R,"src_P":src_P,"src_F":src_F}
        tgt_result = {"epoch": epoch, "valid_R": valid_R, "valid_P": valid_P, "valid_F": valid_F,
                      "test_R": test_R, "test_P": test_P, "test_F": test_F}
        save_prf(result=src_result,name="src_prf",root_path=root_path)
        save_prf(result=tgt_result, name="tgt_prf",root_path=root_path)
        print("source domain: F1=%.4f P=%.4f R=%.4f" % (src_F, src_P, src_R))
        print("target domain'valid dataset: F1=%.4f P=%.4f R=%.4f"%(valid_F,valid_P,valid_R))
        print("target domain'test dataset: F1=%.4f P=%.4f R=%.4f" % (test_F, test_P, test_R))














































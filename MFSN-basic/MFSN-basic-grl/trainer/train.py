import sys
sys.path.append("../")
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_utils import save_model,list_mean,save_prf,save_loss,get_device
import itertools
from functions import DiffLoss

from trainer.evaluate import evaluate


def train(dsn,src_data_loader,tgt_train_data_loader,tgt_valid_data_loader,batch_size,root_path,args):
    print("training!!!!!")
    bestf1 = 0
    # set DSN to "train" model
    dsn.train()
    #loss_function
    ERloss = nn.CrossEntropyLoss()
    Simloss = nn.CrossEntropyLoss()
    Recloss = nn.CrossEntropyLoss()
    Difloss = DiffLoss()
    #拿到device
    device = get_device(args.cuda_id)


    dsn = dsn.to(device)
    ERloss = ERloss.to(device)
    Simloss = Simloss.to(device)
    Recloss = Recloss.to(device)
    Difloss = Difloss.to(device)

    #define optimizer
    optimizer = optim.AdamW(dsn.parameters(), lr=args.lr)
    len_data_loader = min(len(src_data_loader), len(tgt_train_data_loader))


    for epoch in range(args.num_epochs):
        # zip source and target data pair
        #src10，tgt15,那么src采样10条，tgt采样10条，这个方法的缺点：没有使用全部的数据
        data_zip = enumerate(zip(src_data_loader, tgt_train_data_loader))
        #用于统计数据
        src_sim_loss_list = []
        src_dif_loss_list = []
        src_rec_loss_list = []
        src_er_loss_list = []
        tgt_sim_loss_list = []
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



            ###################################
            # target data training            #
            ###################################
            p_loss = 0
            dsn.zero_grad()
            loss = 0
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()
            domain_label = domain_label.to(device)
            #不知道这一步是什么意思
            target_domainv_label = Variable(domain_label)
            if epoch >=args.dann_epoch:
                p = float(step + (epoch - args.dann_epoch) * len_data_loader / (args.num_epochs - args.dann_epoch) / len_data_loader)
                p = 2. / (1. + np.exp(-10 * p)) - 1
                tgt_out_put = dsn(input_ids=tgt_input_ids,attention_mask=tgt_mask_ids,mode="tgt",p=p)
                # compute sim_loss
                tgt_sim_loss = args.sim_weight*Simloss(tgt_out_put["domain_logits"],target_domainv_label)
                loss = loss+tgt_sim_loss

            else:
                tgt_sim_loss = torch.tensor([0])
                tgt_out_put = dsn(input_ids=tgt_input_ids,attention_mask=tgt_mask_ids,mode="tgt")


            #compute diff_loss
            tgt_dif_loss = args.dif_weight*Difloss(tgt_out_put["private_feature"],tgt_out_put["share_feature"])
            loss = loss+tgt_dif_loss


            #compute rec_loss
            tgt_lm_lables = tgt_input_ids[:,1:].contiguous()
            tgt_rec_loss = args.rec_weight*Recloss(tgt_out_put["lm_logits"].view(-1,dsn.config.vocab_size),tgt_lm_lables.view(-1))
            loss = loss + tgt_rec_loss
            p_loss = p_loss+loss.item()

            #用于统计结果

            tgt_sim_loss_list.append(tgt_sim_loss.item())
            tgt_dif_loss_list.append(tgt_dif_loss.item())
            tgt_rec_loss_list.append(tgt_rec_loss.item())

            loss.backward()
            optimizer.step()
            ###################################
            # source data training            #
            ###################################
            dsn.zero_grad()
            loss = 0
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()
            domain_label = domain_label.to(device)
            #不知道这一步是什么意思
            source_domainv_label = Variable(domain_label)

            if epoch >= args.dann_epoch:

                src_out_put = dsn(input_ids=src_input_ids,attention_mask=src_mask_ids,mode="src",p=p)
                # compute sim_loss
                src_sim_loss = args.sim_weight*Simloss(src_out_put["domain_logits"],source_domainv_label)
                loss = loss+src_sim_loss

            else:
                src_sim_loss = torch.tensor([0])
                src_out_put = dsn(input_ids=src_input_ids,attention_mask=src_mask_ids,mode="src")

            #compute diff_loss
            src_dif_loss = args.dif_weight*Difloss(src_out_put["private_feature"],src_out_put["share_feature"])
            loss = loss+src_dif_loss

            #compute rec_loss
            src_lm_lables = src_input_ids[:, 1:].contiguous()
            src_rec_loss = args.rec_weight*Recloss(src_out_put["lm_logits"].view(-1,dsn.config.vocab_size),src_lm_lables.view(-1))
            loss = loss + src_rec_loss

            #compute ER_loss
            src_ER_loss = ERloss(src_out_put["ER_logits"],src_label)
            loss = loss+src_ER_loss
            p_loss = p_loss + loss.item()
            #用于统计结果
            src_sim_loss_list.append(src_sim_loss.item())
            src_dif_loss_list.append(src_dif_loss.item())
            src_rec_loss_list.append(src_rec_loss.item())
            src_er_loss_list.append(src_ER_loss.item())

            loss.backward()
            # optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
            optimizer.step()

            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "Total_loss=%.4f ER_loss=%.4f Sim_loss=%.4f Diff_loss=%.6f Rec_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         p_loss,
                         src_ER_loss.item(),
                         src_sim_loss.item()+tgt_sim_loss.item(),
                         src_dif_loss.item()+tgt_dif_loss.item(),
                         src_rec_loss.item()+tgt_rec_loss.item()
                         ))
            totall_loss_list.append(p_loss)


        #展示每个epoch的loss
        loss_result = {"epoch":epoch,"totall_loss":list_mean(totall_loss_list),"ER_loss":list_mean(src_er_loss_list),
                       "src_sim_loss":list_mean(src_sim_loss_list),"tgt_sim_loss":list_mean(tgt_sim_loss_list),
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
            save_model(net=dsn,name=args.src+"-"+args.tgt+"-basic-grl")
            bestf1 = valid_F

        src_result = {"epoch":epoch,"src_R":src_R,"src_P":src_P,"src_F":src_F}
        tgt_result = {"epoch": epoch, "valid_R": valid_R, "valid_P": valid_P, "valid_F": valid_F,
                      "test_R": test_R, "test_P": test_P, "test_F": test_F}
        save_prf(result=src_result,name="src_prf",root_path=root_path)
        save_prf(result=tgt_result, name="tgt_prf",root_path=root_path)
        print("source domain: F1=%.4f P=%.4f R=%.4f" % (src_F, src_P, src_R))
        print("target domain'valid dataset: F1=%.4f P=%.4f R=%.4f"%(valid_F,valid_P,valid_R))
        print("target domain'test dataset: F1=%.4f P=%.4f R=%.4f" % (test_F, test_P, test_R))














































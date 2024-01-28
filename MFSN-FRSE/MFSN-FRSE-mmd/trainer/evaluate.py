"""Adaptation to train target encoder."""
import torch
import torch.nn as nn
import sys
sys.path.append("..")


def evaluate(model,device,data_loader,mode):
    """Evaluation for encoder and classifier on target dataset."""
    # set eval state for Dropout and BN layer
    model.eval()


    loss = 0
    acc = 0
    tp = 0
    fp = 0
    p = 0
    need_preds = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    count = 0
    # evaluate network
    for (input_ids,mask_ids,seq_len,label) in data_loader:
        input_ids = input_ids.to(device)
        mask_ids = mask_ids.to(device)
        label = label.to(device)


        with torch.no_grad():
            preds = model(input_ids=input_ids,attention_mask=mask_ids,mode=mode)

        pred_cls = preds.data.max(1)[1]

        for i in range(len(label)):
            if label[i] == 1:
                p += 1
                if pred_cls[i] == 1:
                    tp += 1
            else:
                if pred_cls[i] == 1:
                    fp += 1

    div_safe = 0.000001

    recall = tp/(p+div_safe)
    
    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)

    return recall,precision,f1

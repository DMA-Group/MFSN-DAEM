import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def element_add(listA:list,listB:list):
    listC = np.array(listA)+np.array(listB)
    return listC.tolist()


def get_one_loss_figure(ax,x_list,y_list,title_name,label_name):
    ax.plot(x_list, y_list, color="#FF0000", marker="o", markersize="5", lw="0.75", label=label_name)
    ax.set_xlabel("epoch",size=15)
    ax.set_ylabel("loss",size=15)
    ax.set_title(title_name,size=18)
    ax.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0, fontsize="x-small")

def get_loss(dataset,root_path):
    figure, axes = plt.subplots(2, 3, sharex="none", sharey="none", figsize=(25, 10))
    save_path = os.path.join(root_path,"loss_figure.png")
    result_path = os.path.join(root_path,"loss.csv")
    result_df = pd.read_csv(result_path)
    epoch_list = result_df["epoch"].tolist()
    total_loss_list = result_df["totall_loss"].tolist()
    ER_loss_list = result_df["ER_loss"].tolist()
    sim_loss_list = result_df["sim_loss"].tolist()
    dif_loss_list = element_add(listA=result_df["tgt_dif_loss"].tolist(),
                                      listB=result_df["src_dif_loss"].tolist())
    rec_loss_list = element_add(listA=result_df["tgt_rec_loss"].tolist(),
                                      listB=result_df["src_rec_loss"].tolist())
    #toall loss
    get_one_loss_figure(ax=axes[0][0],
                        x_list=epoch_list,y_list=total_loss_list,
                        title_name="total_loss "+" on "+dataset,
                        label_name="total loss")
    #EM loss
    get_one_loss_figure(ax=axes[0][1],
                        x_list=epoch_list,y_list=ER_loss_list,
                        title_name="ER_loss "+" on "+dataset,
                        label_name="ER loss")

    #sim loss
    get_one_loss_figure(ax=axes[0][2],
                        x_list=epoch_list,y_list=sim_loss_list,
                        title_name="sim_loss "+" on "+dataset,
                        label_name="sim loss")

    #dif loss
    get_one_loss_figure(ax=axes[1][0],
                        x_list=epoch_list,y_list=dif_loss_list,
                        title_name="dif_loss "+" on "+dataset,
                        label_name="dif loss")


    #rec loss
    get_one_loss_figure(ax=axes[1][1],
                        x_list=epoch_list,y_list=rec_loss_list,
                        title_name="rec_loss "+" on "+dataset,
                        label_name="rec loss")


    plt.tight_layout()
    plt.delaxes(axes[1][2])
    plt.savefig(save_path)
    plt.clf()



if __name__ == '__main__':

    get_loss(dataset="shoes-cameras")
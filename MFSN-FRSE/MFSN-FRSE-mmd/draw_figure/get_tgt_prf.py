import pandas as pd
import os
import matplotlib.pyplot as plt



def get_tgt_prf(dataset,root_path):
    result_path = os.path.join(root_path,"tgt_prf.csv")
    result_df = pd.read_csv(result_path)
    vaild_f_list = result_df["valid_F"].tolist()
    test_f_list = result_df["test_F"].tolist()
    epoch_list = result_df["epoch"].tolist()

    plt.plot(epoch_list,vaild_f_list,color="#FF0000", marker="o", markersize="5", lw="0.75",label="valid_F1")

    plt.plot(epoch_list, test_f_list, color="#0000FF", marker="^", markersize="5", lw="0.75", label="test_F1")
    plt.xlabel("epoch",size=15)
    plt.ylabel("F1",size=15)
    plt.ylim(0, 1.0)
    plt.title(dataset,size=20)

    plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0, fontsize="x-small")

    plt.tight_layout()
    save_path = os.path.join(root_path,"tgt_prf.png")
    plt.savefig(save_path)
    plt.clf()



if __name__ == '__main__':
    get_tgt_prf(dataset="b2-fz")
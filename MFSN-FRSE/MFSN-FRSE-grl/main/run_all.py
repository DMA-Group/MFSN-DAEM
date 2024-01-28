from main import do_att_GRL
dataset_list1 = [("dzy","fz"),("fz","dzy"),("b2","fz"),("b2","dzy"),("ri","ab"),("ri","wa1"),
                 ("ia","da"),("ia","ds")]
dataset_list2=[("ab","wa1"),("wa1","ab"),("da","ds"),("ds","da")]
dataset_list3= [("shoes","cameras"),("cameras","computers"),("computers","watches"),("watches","shoes")]
dataset_list4= [("watches","computers"),("cameras","watches"),("watches","cameras"),
                ("shoes","watches"),("computers","shoes"),("shoes","computers"),
                ("cameras","shoes"),("computers","cameras")]


for da in dataset_list1+dataset_list3:
    for seed in [22,42,62]:
        do_att_GRL(src=da[0],tgt=da[1],batch_size=32,seed=seed,cuda_id=5)
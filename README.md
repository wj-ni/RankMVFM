#Rank MVFM
Source codes for Rank Multi-View Factorization Machine
#Requirement:
======
Python 3.6  
Tensorflow-gpu 1.10.0

Input data:
======
data/weici/*

Group data:
======
group_config.txt (Save file groups, each line represents a group)

Run the code:
======
```shell
python main.py.py \
  --files_path=data\weici\ \
  --valid_file=coop1_centrality.csv \
  --save_path=save\result.txt \
  --lr=0.01 \
  --epoch=10 \
  --use_l1=False \
  --use_l1=False \
  --use_new_reg=False \
  --l1_reg=0.1 \
  --l2_reg=0.1 \
  --loss=bpr \
  --print=True
```
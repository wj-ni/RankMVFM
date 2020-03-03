###Rank MVFM
Source codes for Rank Multi-View Factorization Machine

###Requirement:
Python 3.6  
Tensorflow-gpu 1.10.0

###Input data:
data/weici/*

###Group data:
group_config.txt (Save file groups, each line represents a group)

### Usage
##### Example Usage
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
  --new_reg=0.1 \
  --loss=bpr \
  --print=True
```
##### Full Command List
```shell
optional arguments:
  --files_path          Original data folder
  --valid_file          Valid filename                   
  --save_path           Result storage path                    
  --lr                  Learning Rate
  --epoch               Training echo
  --use_l1              Use L1 regular or not
  --use_l2              Use L1 regular or not
  --use_new_reg         Use new regular or not
  --l1_reg              L1 regular coefficient                      
  --l2_reg              L2 regular coefficient
  --new_reg             New regular coefficient
  --loss                Loss function(hinge,mean_square,bpr)
  --print               Print results or not
```
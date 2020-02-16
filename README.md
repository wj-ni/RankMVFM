Rank MVFM参数说明

--files_path 	存放数据的文件夹路径 例如data/weici/

--valid_file	选做测试集的文件名，其它文件当训练集 例如coop1_centrality.csv则此文件当测试集，其他文件当训练集

--save_path 	结果保存文件路径，会覆盖原始文件 例如save/test.csv

--hidden_dim 	隐向量维度 默认5

--lr		学习率 默认0.01

--epoch		迭代轮数 默认5

--use_l1		是否使用l1正则 默认False

--use_l2		是否使用l2正则 默认False

--use_new_reg	是否使用新正则 默认False

--l1_reg		l1正则系数 默认0.1

--l2_reg		l2正则系数 默认0.1

--new_reg	新正则系数 默认0.1

--loss		损失函数，只有hinge和mean_square 默认hinge

--print		是否打印结果 默认True

例子python main.py --valid_file=coop1_centrality.csv --save_path=save/test1.csv

group_config.txt存放文件分组情况，每一行代表一个组。更换训练文件时需要配置对应的group_config.txt文件.若不使用组特征，只需将文件中最后一行去掉。
import pandas as pd
import numpy as np
def get_mvfm_data(filepath,view_list):
    view_list=[num-1 for num in view_list]
    res=[]
    for i in range(len(view_list)):
        res.append(sum(view_list[:i+1]))
    df=pd.read_csv(filepath)

    feature_list=get_group('group_config.txt')
    df=df[feature_list]

    groupByNew =df.groupby('typeId')
    for name, groupData in groupByNew:
        if name==1:
            gropu1=groupData
        elif name==2:
            group2=groupData
        else:
            group0=groupData
    # pos sample
    first_list = list(group2)
    first_list.remove('typeId')
    first_sample = group2[first_list]
    first_sample = first_sample.values
    first_sample = np.insert(first_sample, res, 1, axis=1)
    #pos sample
    # pos_sample=pd.concat([gropu1,group2],axis=0)
    pos_list=list(gropu1)
    pos_list.remove('typeId')
    pos_sample=gropu1[pos_list]
    pos_sample=pos_sample.values
    pos_sample=np.insert(pos_sample,res,1,axis=1)

    #neg sample
    neg_list=list(group0)
    neg_list.remove('typeId')
    neg_sample=group0[neg_list]
    neg_sample=neg_sample.values
    neg_sample=np.insert(neg_sample,res,1,axis=1)

    return first_sample,pos_sample,neg_sample

def get_mvfm_data_mean_square(filepath,view_list):
    view_list=[num-1 for num in view_list]
    res=[]
    for i in range(len(view_list)):
        res.append(sum(view_list[:i+1]))
    df=pd.read_csv(filepath)

    feature_list = get_group('group_config.txt')
    df = df[feature_list]

    label=df.typeId
    labels=label.values

    df_feature=list(df)
    df_feature.remove('typeId')
    df=df[df_feature]
    values=df.values
    values=np.insert(values, res, 1, axis=1)
    return values,labels
def get_view_list(file):
    view_list=[]
    with open(file,encoding='utf-8') as f:
        for line in f.readlines():
            lineArr=line.strip().split(',')
            view_list.append(len(lineArr))
    return view_list
def get_group(file):
    group=['typeId']
    with open(file,encoding='utf-8') as f:
        for line in f.readlines():
            lineArr=line.strip().split(',')
            group.extend(lineArr)
    return group
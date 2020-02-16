import tensorflow as tf
from model.mvfm_hig_consta_hinge_loss import mvFm as mvFm_hinge
from model.mvfm_hig_consta_mean_square_loss import mvFm as mvFm_mean_square
from util.data import get_mvfm_data,get_mvfm_data_mean_square,get_view_list
from util.metric import ndcg_score
import pandas as pd
import argparse
import os

class train_mvfm():
    def __init__(self,file_path,valid_file,view_list,loss='hinge'):
        assert loss in ['hinge','mean_square'],'loss error'
        self.file_path=file_path
        self.files=os.listdir(file_path)
        self.valid_file=valid_file
        assert self.valid_file in self.files,'valid_file does not exist'
        self.view_list=[num+1 for num in view_list]
        self.loss=loss
    def train( self ,dim,learning_rate,epoch,scope_name,use_l1=False,use_l2=False,use_new_reg=False,l1_reg=0.1,l2_reg=0.1,new_reg=0.1):
        with tf.Session() as sess:
            with tf.variable_scope(scope_name):
                if self.loss=='hinge':
                    model = mvFm_hinge(self.view_list, dim, learning_rate, use_l1,use_l2,use_new_reg,l1_reg,l2_reg,new_reg)
                elif self.loss=='mean_square':
                    model=mvFm_mean_square(self.view_list, dim, learning_rate, use_l1,use_l2,use_new_reg,l1_reg,l2_reg,new_reg)
            sess.run(tf.global_variables_initializer())
            res=[]
            file_name='normal_distribution_train'
            # file_name = 'normal_distribution_nottrain'
            # file_name = 'one_distribution_train'
            # file_name = 'one_distribution_nottrain'
            with open(file_name+'_loss.txt','w',encoding='utf-8') as f_loss,open(file_name+'_grad.txt','w',encoding='utf-8') as f_grad:
                for n in range(epoch):
                    loss_sum=0
                    n=0
                    grad_sum=None
                    for file in self.files:
                        if file == self.valid_file:
                            continue
                        else:
                            if self.loss=='hinge':
                                first_sample, pos_sample, neg_sample = get_mvfm_data(self.file_path + file, self.view_list)
                                loss,grad, _ = sess.run([model.loss,model.grad, model.train_op],feed_dict={model.first_sample: first_sample, model.pos_sample: pos_sample,model.neg_sample: neg_sample})
                                loss_sum+=loss
                                n+=1
                                grad=grad[0]
                                if grad_sum is None:
                                    grad_sum=grad
                                else:
                                    grad_sum+=grad
                            elif self.loss=='mean_square':
                                sample, target = get_mvfm_data_mean_square(self.file_path + file, self.view_list)
                                loss, grad,_ = sess.run([model.loss,model.grad, model.train_op],feed_dict={model.sample: sample, model.target: target})
                                loss_sum+=loss
                                n+=1
                                grad=grad[0]
                                if grad_sum is None:
                                    grad_sum=grad
                                else:
                                    grad_sum+=grad
                    # print(loss_sum/n)
                    f_loss.write(str(loss_sum/n)+'\n')
                    grad_sum=grad_sum.reshape(-1).tolist()
                    for num in grad_sum:
                        f_grad.write(str(num)+' ')
                    f_grad.write('\n')
                    if self.loss=='hinge':
                        res.append(self.save_ndcg_hinge(sess, model))
                    elif self.loss=='mean_square':
                        res.append(self.save_ndcg_mean_square(sess,model))
            return res
    def save_ndcg_hinge( self,sess, model ):
        first_sample, pos_sample, neg_sample = get_mvfm_data(self.file_path + self.valid_file, self.view_list)
        first, pos, neg = sess.run([model.first, model.pos, model.neg],feed_dict={model.first_sample: first_sample, model.pos_sample: pos_sample, model.neg_sample: neg_sample})
        first = list(first)
        first_len = len(first)
        pos = list(pos)
        pos_len = len(pos)
        neg = list(neg)
        neg_len = len(neg)
        neg.extend(pos)
        neg.extend(first)
        first_tag = [2] * first_len
        pos_tag = [1] * pos_len
        neg_tag = [0] * neg_len
        neg_tag.extend(pos_tag)
        neg_tag.extend(first_tag)
        res = []
        for n in range(10):
            res.append(ndcg_score(neg_tag, neg, n + 1))
        return res
    def save_ndcg_mean_square( self,sess, model ):
        sample, target = get_mvfm_data_mean_square(self.file_path + self.valid_file, self.view_list)
        pred = sess.run(model.pred, feed_dict={model.sample: sample, model.target: target})
        res = []
        for n in range(10):
            res.append(ndcg_score(target, list(pred), n + 1))
        return res
def main(files_path,valid_file,save_path,view_list,hidden_dim,lr,use_l1,use_l2,use_new_reg,l1_reg,l2_reg,new_reg,epoch,loss='mean_square',print_result=True):
    if os.path.exists('/'.join(save_path.split('/')[:-1])) is False:
        os.makedirs('/'.join(save_path.split('/')[:-1]))
    t_mvfm=train_mvfm(files_path,valid_file,view_list,loss)
    res=t_mvfm.train(hidden_dim,lr,epoch,valid_file+str(lr),use_l1,use_l2,use_new_reg,l1_reg,l2_reg,new_reg)
    df=pd.DataFrame(res,columns=['NDCG@'+str(i+1) for i in range(len(res[0]))])
    df.to_csv(save_path,index=False)
    if print_result:
        print('NDCG@n',res)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", help="all files path", type=str,default='data/weici/')
    parser.add_argument("--valid_file", help="test file name", type=str,default='coop2_centrality.csv')
    parser.add_argument("--save_path", help="result save path", type=str,default='save/result.txt')
    parser.add_argument("--hidden_dim", help="hidden vector dim", type=int,default=5)
    parser.add_argument("--lr", help="learning rate", type=float,default=0.01)
    parser.add_argument("--epoch", help="epoch", type=int,default=5)
    parser.add_argument("--use_l1", help="bool use l1 reg or not", type=bool, default=False)
    parser.add_argument("--use_l2", help="bool use l2 reg or not", type=bool, default=False)
    parser.add_argument("--use_new_reg", help="bool use new reg or not", type=bool, default=False)
    parser.add_argument("--l1_reg", help="l1 reg number", type=float, default=0.1)
    parser.add_argument("--l2_reg", help="l2 reg number", type=float, default=0.1)
    parser.add_argument("--new_reg", help="new reg number", type=float, default=0.1)
    parser.add_argument("--loss",help="loss type,hinge or mean_square",type=str,default='mean_square')
    parser.add_argument("--print",help='Whether to print output',type=bool,default=True)
    args = parser.parse_args()
    view_list=get_view_list('group_config.txt')
    assert args.hidden_dim>0,'hidden_dim should >0'
    assert args.lr>0,'learning rate should >0'
    assert args.epoch>0,'epoch should >0'
    assert args.l1_reg>0,'l1 reg shuold >0'
    assert args.l2_reg>0,'l2 reg should >0'
    assert args.new_reg>0,'reg should >0'
    main(args.files_path,args.valid_file,args.save_path,view_list,args.hidden_dim,args.lr,args.use_l1,args.use_l2,args.use_new_reg,args.l1_reg,args.l2_reg,args.new_reg,args.epoch,args.loss,args.print)
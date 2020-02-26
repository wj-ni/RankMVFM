import tensorflow as tf
class mvFm:
    def __init__(self,view_list,dim,learning_rate,use_l1=False,use_l2=False,use_new_reg=False,l1_reg=0.1,l2_reg=0.1,new_reg=0.1):
        for node in view_list:
            assert type(node) is int,'view_list 数据类型必须是整数'
            assert node>0,'view_list的值必须大于0'
        self.dim=dim
        self.learning_rate=learning_rate
        self.view_size=len(view_list)
        self.view_sum=0
        for node in view_list:
            self.view_sum+=node
        self.view_list=[node-1 for node in view_list]
        self.create_placehold()
        with tf.variable_scope('sample'):
            self.pred=self.cal_mvfm(self.sample,view_list)
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.target, self.pred,reduction=tf.losses.Reduction.MEAN)
            if use_l1:
                self.loss+=tf.contrib.layers.l1_regularizer(l1_reg)(self.var_a)
            if use_l2:
                self.loss += tf.contrib.layers.l2_regularizer(l2_reg)(self.var_a)
            if use_new_reg:
                self.loss+=new_reg*self.cal_reg(view_list)
        with tf.variable_scope('train'):
            optimizer_adam = tf.train.AdamOptimizer(self.learning_rate)
            # self.grad=optimizer_adam.compute_gradients(self.loss,aggregation_method=tf.AggregationMethod.ADD_N)
            self.grad=tf.gradients(self.loss,self.var_a)
            self.train_op=optimizer_adam.minimize(self.loss)
    def create_placehold(self):
        self.sample=tf.placeholder('float32',[None,self.view_sum],name='first')
        self.target=tf.placeholder('float32',[None],name='target')
        self.var_a=tf.get_variable('a',dtype=tf.float32,shape=[self.view_sum-len(self.view_list),self.dim],initializer=tf.variance_scaling_initializer(seed=1))
        # atten=tf.get_variable('atten',dtype=tf.float32,shape=[self.view_sum-len(self.view_list),1],initializer=tf.variance_scaling_initializer(seed=1))
        # self.var_a=tf.multiply(self.var_a,atten)
        self.const_a=tf.get_variable('const_a',dtype=tf.float32,shape=[len(self.view_list),self.dim],initializer=tf.variance_scaling_initializer(seed=1),trainable=False)
        self.temp_a=tf.split(self.var_a,self.view_list,axis=0)
        self.temp_const_a=tf.split(self.const_a,[1]*len(self.view_list),axis=0)
        a_list=[]
        for a,const_a in zip(self.temp_a,self.temp_const_a):
            a_list.append(a)
            a_list.append(const_a)
        self.a=tf.concat(a_list,axis=0)

    def cal_mvfm( self,sample,view_list ):
        # pos sample
        z_list = tf.split(sample, view_list, axis=1)  # [bs,view_size]
        a_list = tf.split(self.a, view_list, axis=0)  # [view_size,dim]

        z_a_dot_sum = []
        for z, a in zip(z_list, a_list):
            # 求z和a的点积
            z_expand = tf.expand_dims(z, dim=-1)  # [bs,view_size,1]
            a_expand = tf.expand_dims(a, dim=0)  # [1,view_size,dim]
            temp = tf.multiply(z_expand, a_expand)  # bs,v_size,dim
            # 纵向求和
            temp = tf.reduce_sum(temp, axis=1)  # bs,dim
            z_a_dot_sum.append(temp)
        # 多个向量连续点积
        dot_res = None
        for node in z_a_dot_sum:
            if dot_res is None:
                dot_res = node
            else:
                dot_res = tf.multiply(dot_res, node)  # bs,dim
        y = tf.reduce_sum(dot_res, axis=-1)  # bs
        return y

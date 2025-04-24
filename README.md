# Transformer_code
此处是学习笔记
代码来源于b站up主：hayyp魇
链接：https://www.bilibili.com/video/BV1Fw4m1C7Tq/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=0234fe0a88066b7533478f310d4a7e05
## 多头注意力机制
''' python
    import torch
    from torch import nn
    import torch.functional as F
    import math
    
    X=torch.randn(128,64,512) #Batch Time Dimension
    print(X.shape)
    d_model=512
    n_head=8
    
    class multi_head_attention(nn.Module):
        def __init__(self,d_model,n_head)->None:
            '''
            初始化多头注意力模块
            :param d_model: 输入向量的总维度（此处是常用的512维）
            :param n_head: 注意力头的数量
            '''
            super(multi_head_attention,self).__init__()
            #存储模型参数
            self.d_model=d_model    #输入总维度
            self.n_head=n_head      #注意力头数
    
            #线性层映射函数，将初始的向量映射到QKV
            self.w_q=nn.Linear(d_model,d_model)
            self.w_k=nn.Linear(d_model,d_model)
            self.w_v=nn.Linear(d_model,d_model)
            #输出合并层
            self.w_combine=nn.Linear(d_model,d_model)
            #按最后一个维度做一个归一化
            self.softmax=nn.Softmax(dim=-1)
    
        def forward(self,q,k,v):
            '''
            前向传播过程
            :param q: 查询向量
            :param k: 键向量
            :param v: 值向量
            :return:
            '''
            batch,time,dimension=q.shape
            n_d=self.d_model//self.n_head   #计算每个头的维度512/8
            #线性投影
            q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
            #重塑形状为 [batch, time, n_head, n_d]，再转置为 [batch, n_head, time, n_d]
            q=q.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
            k=k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
            v=v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
            #注意力计算（缩放点积注意力）     @是矩阵乘法
            score=q@k.transpose(2,3)/math.sqrt(n_d)
            #因果掩码
            mask=torch.tril(torch.ones(time,time,dtype=bool))  #下三角矩阵
            score=score.masked_fill(mask==0,float('-inf'))     #将0填充为负无穷
            #softmax归一化与加权求和
            score=self.softmax(score)@v
            #多头输出合并
            score=score.permute(0,2,1,3).contiguous().view(batch,time,dimension) ## 转回 [batch, time, n_head, n_d]
            output=self.w_combine(score) ## 合并为 [batch, time, d_model]
            return output
    #实例化调用
    attention=multi_head_attention(d_model,n_head)
    output=attention(X,X,X)    #自注意力模式Q=K=V
    print(output,output.shape)

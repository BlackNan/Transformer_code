# Transformer_code
此处是自用学习笔记，每行代码加了注释和部分函数的用法
代码来源于b站up主：hayyp魇  

链接：https://www.bilibili.com/video/BV1Fw4m1C7Tq/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=0234fe0a88066b7533478f310d4a7e05  

## 多头注意力机制
```python 
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
```
### 因果掩码
`mask=torch.tril(torch.ones(time,time,dtype=bool))`  #下三角矩阵
 `score=score.masked_fill(mask==0,float('-inf'))`     #将0填充为负无穷
 - **核心目的**：在自回归生成任务（如文本生成、机器翻译）中，模型预测第`i`个位置的输出时，**只能依赖已生成的`1~i-1`位置信息**，而不能“偷看”未来的`i+1~n`位置。这是生成任务的基本逻辑要求。
- **实现原理**：下三角矩阵（`tril`）的对角线及以下为`True`（允许关注），以上为`False`（屏蔽未来）。通过将`mask == 0`（即未来位置）的注意力分数设为负无穷，使得这些位置在后续`softmax`计算中权重趋近于零

因果注意力的工作原理是通过掩码矩阵限制模型在计算每个时间步的注意力时，只关注当前时间步及之前的内容。具体地，掩码矩阵是一个上三角矩阵，其上三角部分为0，其余部分为1。这样，在计算注意力分布时，掩码矩阵将未来时间步的注意力得分设置为非常大的负值（`-inf`），使得这些位置在 `softmax` 操作后接近于零，从而不会对最终的输出产生影响。

掩码矩阵的结构如下：

```
[
 [1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]
]
```

## `masked_fill` 函数

`Tensor.masked_fill(mask, value) → Tensor`
- **参数**：
    - `mask`：布尔类型张量（`torch.BoolTensor`），形状需与原张量**可广播**（broadcastable）
    - `value`：替换值（标量，如`0`、`inf`、`1e9`等）
- **功能**：将原张量中`mask`为`True`的位置替换为`value`，其余位置保留原值。

## QKV的线性变换

在Transformer模型中，对输入向量分别进行Q（Query）、K（Key）、V（Value）的线性变换是设计中的核心机制，其根本原因在于**通过参数化投影赋予模型动态学习不同语义角色的能力**。

- 线性变换为Q/K/V引入可训练参数，扩展了模型对不同输入分布的适应能力（实验表明，去除线性变换会导致模型无法捕捉复杂依赖）
- 线性变换将输入向量拆分到多个子空间（头），实现并行化多视角计算

# 位置编码position embedding

```python
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,maxlen,device):
        '''
        定义继承自nn.Module的位置编码类。
        :param d_model:嵌入向量的维度（transformer常用维度为512）
        :param maxlen:序列的最大长度
        :param device:指定张量存储的设备（CPU或GPU）
        '''
        super(PositionalEmbedding,self).__init__()
        #初始化位置编码矩阵，创建形状为(maxlen,d_model)的全零矩阵，表示位置编码表
        self.encoding=torch.zeros(maxlen,d_model,device)
        #位置编码是静态的，不参与梯度更新
        self.encoding.requires_grad=False

        #生成一维位置序列（0-maxlen）,并扩展为二维矩阵
        pos=torch.arange(0,maxlen,device)  #[0,1,2,3,...]
        #将位置索引转换为列向量,形状变为(maxlen,1)
        pos=pos.float().unsqueeze(1)      #[[0],[1],[2],[3],...]
        #生成步长为2的索引序列（0, 2, 4, ..., d_model-2）
        _2i=torch.arange(0,d_model,2,device)
        #按奇数列和偶数列分别填充正弦和余弦值（不同频率的正弦/余弦函数能捕捉序列中词与词之间的相对位置关系）
        self.encoding[:,0::2]=torch.sin(pos/10000**(_2i/d_model))
        self.encoding[:,1::2]=torch.cos(pos/10000**(_2i/d_model))

        def forward(self, x):
            '''

            :param x:x的形状为 (batch_size, seq_len)
            :return:(seq_len, d_model) 的位置编码矩阵
            '''
            #获取输入序列长度
            seq_len=x.shape[1]
            #[:seq_len]:第0-seq_len-1 行的切片,第二个冒号 : 表示保留所有列（d_model 维度全选）
            return self.encoding[:seq_len,:]
```

## 位置矩阵

$$
PE_{(pos,2i)} = \sin\left( \frac{pos}{10000^{2i/d}} \right)
$$

$$
PE_{(pos,2i+1)} = \cos\left( \frac{pos}{10000^{2i/d}} \right)
$$

矩阵的行表示位置（0到maxlen-1），列表示不同频率的正弦/余弦分量

- 矩阵的每一行对应序列中的一个具体位置索引（从0到`maxlen-1`）
- 每一列对应不同的频率分量，频率由公式中的分母项 `10000^(2i/d_model)` 控制

（这样设计的原因在于它具有周期性，可以帮助模型处理比训练时更长的序列，同时保持一定的泛化能力）

- **高频分量**（小波长）：当`i`较小时，分母项`10000^(2i/d_model)`较小，频率较高，适合捕捉局部位置关系（如相邻词的位置差异）。
- **低频分量**（大波长）：当`i`较大时，分母项较大，频率较低，适合捕捉长距离位置关系（如段落或文档级别的结构）。

# token embedding

```python
class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        '''
        将离散的词汇索引映射为稠密的词向量表示
        :param vocab_size:词表大小
        :param d_model:词向量维度
        padding_idx=1（填充索引）：指定索引1为填充符（如<PAD>），该位置的嵌入向量在训练过程中不会更新，且默认初始化为零
        '''
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)
```

`nn.Embedding`本质上是一个可训练的查询矩阵（形状为`[num_embeddings, embedding_dim]`），每个索引（如单词ID）通过矩阵查找得到对应的向量。

将输入的词汇索引（如句子中的单词ID）转换为连续的稠密向量。例如，输入`[2, 5, 1]`（其中`1`是填充符）会被映射为形状`[3, d_model]`的矩阵。

输入：词汇索引→输出：对应词向量

### **底层实现原理**

1. **权重矩阵**：`TokenEmbedding`内部维护一个形状为`[vocab_size, d_model]`的权重矩阵，初始值服从随机分布（可通过预训练权重加载）。
2. **查询过程**：输入索引通过One-Hot编码与权重矩阵相乘，等效于直接取权重矩阵的对应行
3. **梯度计算**：填充符（`padding_idx`）对应的行不参与梯度更新，从而避免无效计算

[什么是词向量？如何实现词向量？](https://mp.weixin.qq.com/s?__biz=MzA3MjkzMjYxNQ==&mid=2457824963&idx=1&sn=0d53debd0eecbdaf2110cb9d52337ef0&chksm=89b596a556380b0afc95b4014186a75dbf32a936b75fd15b53e558deee9af0a5f3ac46ac230c#rd)

# Total Embedding

```python
class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(vocab_size,d_model)
        self.pos_emb=PositionalEmbedding(d_model,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)
        
        def forward(self, x):
            tok_emb=self.tok_emb(x)
            pos_emb=self.pos_emb(x)
            return self.drop_out(tok_emb+pos_emb)
```

# LayerNorm

LayerNorm是用来对神经网络的激活值进行归一化的技术，和BatchNorm不同，它是在特征维度上做的归一化，适用于序列模型如Transformer。Transformer通常在每个子层（如多头注意力、前馈网络）后接LayerNorm，形成残差连接结构。

---

### **为什么需要LayerNorm？**

### **1. 解决内部协变量偏移（Internal Covariate Shift）**

在深度神经网络中，随着参数更新，每一层的输入分布会逐渐偏离初始状态（即“协变量偏移”），导致后续层的参数需要不断适应这种变化，从而降低训练效率。LayerNorm 通过**对每个样本的特征维度进行归一化**，将输入重新调整为均值为0、方差为1的分布，从而缓解这一问题，加速收敛。

### **2. 适配序列数据与动态输入**

- **序列长度可变性**：在自然语言处理（NLP）中，输入序列（如句子）的长度通常是动态变化的。BatchNorm 依赖固定批次的统计量，而 LayerNorm **独立处理每个样本**，无需跨样本计算，因此天然适配变长输入。
- **小批量或单样本训练**：当批量大小较小（如在线学习）或动态变化时，BatchNorm 的统计量可能不准确，而 LayerNorm 的稳定性更高。

### **3. 提升模型泛化能力**

LayerNorm 通过可学习的缩放（`gamma`）和平移（`beta`）参数，允许模型在归一化后恢复部分原始分布特征，增强表达能力。这种设计既能保留特征间的相对关系，又能避免归一化导致的信息损失

### **4. 与Transformer架构的兼容性**

---

在 Transformer 中，LayerNorm 被广泛用于每个子层（如多头注意力、前馈网络）的输出后，结合残差连接，有效解决了深层网络中的梯度消失/爆炸问题，成为模型稳定训练的关键组件

### **归一化公式**

```python

output = (x - mean) / torch.sqrt(var + self.eps)
```

- **数学公式**：

- 其中，`μ`是均值，`σ²`是方差，`ε`为`eps`。

$$
\hat{x}_i = \gamma \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

- **效果**：将输入数据转换为均值为0、方差为1的分布，消除特征间的量纲差异。
- **可学习参数：***γ*（缩放）和*β*（偏移）允许模型恢复归一化可能丢失的原始分布信息，增强表达能力
- **维度选择**：计算均值和方差时，**沿特征维度（而非批次或序列维度）**，例如输入形状为`(batch_size, seq_len, d_model)`时，对每个样本的`d_model`维度独立计算。

---

### **二、LayerNorm与BatchNorm的核心差异**

### **1. 归一化维度**

| **特性** | **LayerNorm** | **BatchNorm** |
| --- | --- | --- |
| **归一化范围** | 单个样本的所有特征维度 | 同一特征在批次内的所有样本 |
| **计算维度** | 沿特征维度（如`d_model`）计算均值和方差 | 沿批次维度计算每个通道的均值和方差 |
| **输入形状示例** | `(batch_size, seq_len, d_model)` | `(batch_size, channels, height, width)` |
- **LayerNorm**：适用于序列数据（NLP、语音），对特征维度归一化。
- **BatchNorm**：适用于图像数据（CNN），对通道维度归一化。
![image](https://github.com/user-attachments/assets/bb5cfb0b-3d77-432f-b205-191dba5f2bf1)



`C为通道维d_model，N为batch`

---

### **典型应用场景**

| **场景** | **推荐方法** | **原因** |
| --- | --- | --- |
| 图像分类（CNN） | BatchNorm | 利用批次统计，加速训练并稳定梯度 |
| Transformer（NLP） | LayerNorm | 适配变长序列，无需依赖批次大小 |
| 小批量/动态输入 | LayerNorm | 统计量稳定，适合在线学习 |
| 风格迁移（GAN） | InstanceNorm | 对单样本的通道独立归一化 |

---

```python
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-10):
        '''
        通过继承PyTorch的nn.Module类，可以复用其参数管理和自动梯度计算功能
        :param d_model:特征维度的大小，即输入张量的最后一个维度（例如词向量的维度）
        :param eps:防止分母为零的小常数，默认值为1e-10
        '''
        super(LayerNorm,self).__init__()
        #直接归一化会破坏原始数据的分布，引入可学习参数使模型能自适应调整输出分布
        self.gamma = nn.Parameter(torch.ones(d_model))  #对归一化后的数据进行缩放，初始化为全1
        self.beta = nn.Parameter(torch.zeros(d_model))  #对归一化后的数据进行平移，初始化为全0
        self.eps=eps    #当方差接近零时，添加eps避免除零错误

        def forward(self, x):
            #沿输入张量的最后一个维度（即特征维度）计算均值和方差
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1,unbiased=False, keepdim=True)    #unbiased=False：使用有偏方差计算（分母为n而非n-1）
            #归一化公式
            output = (x - mean) / (torch.sqrt(var + self.eps))
            #缩放与偏移
            output = self.gamma * output + self.beta
            return output
```

## **位置感知前馈网络（Positionwise Feed-Forward Network，FFN）**

其核心功能是通过两次线性变换和非线性激活增强模型表达能力。

```python
class PositionwiseFeedForward(nn.Module):
    '''
    该代码实现了Transformer中的位置感知前馈网络（Positionwise Feed-Forward Network，FFN），其核心功能是通过两次线性变换和非线性激活增强模型表达能力。
    '''
    def __init__(self,d_model,hidden,dropout=0.1):
        '''
        通过fc1将输入从d_model升维至hidden，再通过fc2降回d_model，形成d_model → hidden → d_model的沙漏结构
        :param d_model: 模型主维度
        :param hidden: 模型扩展维度
        :param dropout:
        '''
        self.fc1=nn.Linear(d_model,hidden)  # 输入维度→扩展维度
        self.fc2=nn.Linear(hidden,d_model)  # 扩展维度→还原维度
        self.dropout=nn.Dropout(dropout)
        def forward(self, x):
            x=self.fc1(x)   #升维：拓展特征空间
            x=F.relu(x)     #引入非线性
            x=self.dropout(x)   #正则化：屏蔽部分神经元防止过拟合
            x=self.fc2(x)   #降维：浓缩特征信息
            return x
```

![image](https://github.com/user-attachments/assets/c0034a5a-79b7-471b-96ec-80398bb7f731)


# Transfomer Encoder Layer

![image](https://github.com/user-attachments/assets/f8658b8d-ca49-485f-af9a-3e5c74791981)


```python
class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob)->None:
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.drop1=nn.Dropout(p=drop_prob)

        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm2=LayerNorm(d_model)
        self.drop2=nn.Dropout(p=drop_prob)
    def forward(self,x,mask=None):
        # 保存原始输入用于残差连接（子层1）
        _x=x
        x=self.attention(x,x,x,mask)    # 自注意力计算(Q=K=V=x)

        x=self.drop1(x)     # 随机失活
        self.norm1(x+_x)    # 残差连接 + 层归一化

        # 保存子层1输出用于残差连接（子层2）
        _x=x
        x=self.ffn(x)   # 前馈网络处理

        self.drop2(x)   # 正则化
        x=self.norm2(x+_x)  # 残差连接 + 层归一化

        return x
```

### **残差连接的作用**

- **梯度稳定**：通过跳跃连接绕过深层网络，防止梯度消失（Vanishing Gradient）
- **信息无损传递**：确保深层网络不会丢失原始输入特征
- **收敛加速**：相比纯堆叠网络，残差结构使训练速度提升2-3倍

# Transfomer Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(p=drop_prob)

        self.cross_attention=MultiHeadAttention(d_model,n_head)
        self.norm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(p=drop_prob)

        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm3=LayerNorm(d_model)
        self.dropout3=nn.Dropout(p=drop_prob)

        def forward(self,dec,enc,t_mask,s_mask):
            _x=dec
            x=self.attention(dec,dec,dec,t_mask) #t_mask下三角掩码

            x=self.dropout1(x)
            x=self.norm1(x+_x)

            if enc is not None:
                _x=self.cross_attention(x,enc,enc,s_mask) #s_mask位置的掩码

                x=self.dropout2(x)
                x=self.norm2(x+_x)

            _x=x
            x=self.ffn(x)

            x=self.dropout3(x)
            x=self.norm3(x+_x)
            return x
```

# Encoder

```python
class Encoder(nn.Module):
    def __init__(self,env_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        '''
        :param dec_voc_size:输入词汇表大小（如自然语言中的词表总数）
        :param max_len:输入序列的最大长度，用于位置编码（Positional Encoding）
        :param d_model:模型的隐藏层维度（如 512 或 1024）
        :param ffn_hidden:前馈网络（FFN）的隐藏层维度
        :param n_head:多头注意力机制的头数
        :param n_layer:编码器层的堆叠次数
        :param drop_prob:Dropout 概率，用于正则化
        :param device:计算设备（如 CPU 或 GPU）
        '''
        super(Encoder,self).__init__()
        self.embedding=TransformerEmbedding(d_model,max_len,env_voc_size,drop_prob,device)  #输入序列的离散词索引转换为连续向量，并加入位置信息
        self.layers=nn.ModuleList(
            [EncoderLayer(d_model,ffn_hidden,n_head,drop_prob) for _ in range(n_layer)])    #堆叠 n_layer 个编码器层
        def forward(self,x,s_mask):
            x=self.embedding(x)
            for layer in self.layers:
                x=layer(x,s_mask)
            return x
```

**`for _ in range(n_layer)`** 

用于创建多个独立的**编码器层（EncoderLayer）**，堆叠`n_layer`次。每个层结构相同但参数独立，通过多次堆叠提升模型对复杂模式的捕捉能力。

`_` 是 Python 中的惯用符号，表示循环变量未被使用（仅需循环次数）
例如，若 `n_layer=6`，则会创建 6 个 `EncoderLayer`

# Decoder

```python
class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Decoder,self).__init__()
        self.embedding=TransformerEmbedding(d_model,max_len,dec_voc_size,drop_prob,device)
        self.layers=nn.ModuleList(
            [EncoderLayer(d_model,ffn_hidden,n_head,drop_prob) for _ in range(n_layer)])
        self.fc=nn.Linear(d_model,dec_voc_size)
        def forward(self,dec,enc,t_mask,s_mask):
            dec=self.embedding(dec)
            for layer in self.layers:
                dec=layer(dec,enc,t_mask,s_mask)
            dec=self.fc(dec)
            return dec
```

# Transformer

## ne函数

`torch.ne()` 是 PyTorch 中用于比较张量元素是否不相等的函数，其名称来源于“**not equal**”。
**作用**：比较两个张量中对应位置的元素是否不相等，返回布尔类型张量（`True`表示不相等，`False`表示相等）

```python
a = torch.tensor([1, 2, 3])
result = torch.ne(a, 2)# 输出：tensor([True, False, True])
```

此时，结果为布尔张量，标记哪些位置不等于。

---

## 解码器自注意力mask

在Transformer解码器的自注意力层中，将Padding Mask和Causal Mask相乘（即逻辑与操作）是为了**同时解决两种关键限制**：既要忽略无效的填充符，又要防止模型看到未来信息。

- **Padding Mask的生成**
    
    通过判断每个位置是否为填充符（例如`q.ne(pad_idx)`），生成布尔矩阵：
    
    - `True`：有效位置（非填充符）
    - `False`：无效位置（填充符）。
- **Causal Mask的生成**
    
    使用下三角矩阵（`torch.tril`），将未来位置设为`False`，过去和当前位置设为`True`
    

布尔值的乘法**等价于逻辑与操作**：只有当两个掩码的对应位置均为`True`时，结果才为`True`
```python
class Transformer(nn.Module):
    def __init__(self,src_pad_idx,trg_pad_idx,
                 enc_voc_size,dec_voc_size,
                 max_len,d_model,n_heads,ffn_hidden,n_layers,drop_prob,device):
        super(Transformer,self).__init__()

        self.encoder=Encoder(enc_voc_size,max_len,d_model,ffn_hidden,n_heads,n_layers,drop_prob,device)
        self.decoder=Decoder(dec_voc_size,max_len,d_model,ffn_hidden,n_heads,n_layers,drop_prob,device)

        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device

    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        '''
        屏蔽填充符,适配注意力计算
        逻辑与（&）：确保只有当q的第i位置和k的第j位置均为有效时，mask[i][j]才为True
        '''
        len_q,len_k=q.size(1),k.size(1)

        #(Batch,Time,len_q,len_k)
        # 生成q的布尔掩码 (标记非填充位置)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3) # 扩展为(batch,seq_q)
        q=q.repeat(1,1,1,len_k)                     # 扩展为(batch, 1, seq_q, 1)
        # 生成k的布尔掩码 (标记非填充位置)
        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2) # (batch,seq_k)
        k=k.repeat(1,1,len_k,1)                     # 扩展为(batch, 1, 1, seq_k)
        # 逻辑与操作：仅当q和k均为非填充时，位置有效
        mask=q&k
        return mask

    def make_casual_mask(self,q,k):
        '''
        torch.tril()生成下三角矩阵（左下角为True，右上角为False），允许当前位置关注过去及自身
        '''
        len_q,len_k = q.size(1),k.size(1)
        mask=torch.tril(torch.ones((len_q,len_k)).type(torch.BoolTensor).to(self.device))
        return mask

    def forward(self,src,trg):
        ## 编码器：源序列自注意力掩码
        src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        # 解码器：目标序列自注意力掩码（Padding + Causal）
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx)*self.make_casual_mask(trg,trg) #填充部分不被关注+未来信息不被看到
        # 解码器-编码器交叉注意力掩码（目标与源序列对齐）
        src_trg_mask=self.make_pad_mask(trg,src,self.trg_pad_idx,self.src_pad_idx)

        enc=self.encoder(src,src_mask)
        output=self.decoder(trg,enc,trg_mask,src_trg_mask)
        return output
```

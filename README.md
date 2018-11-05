# 201814844YangMiao
Data Mining

### HomeWork1

#### 运行代码

~~~python
python -u VSM_KNN.py > output.log
~~~

#### 实验内容

+ 预处理文本数据集，并且得到每个文本的VSM表示。
+ 实现KNN分类器，测试其在20Newsgroups上的效果。

#### 实验环境

- Windows 8.1
- Python 3.6(Anaconda)
- CPU-only

#### 实验内容

##### VSM模型

+ 使用textblob package进行数据预处理。包括：Normalization、Tokenization、Stemming、Stopwords。
+ 建立每个文件的字典，计算Term Frequency。
+ 均匀从每个类中采样，选取80%训练集，20%测试集合。
+ 根据训练集字典建立Global dictionary同时计算Document Frequency。
+ 利用TF-IDF方法计算每个Term的Weight，并据此建立Vector。

##### KNN模型

+ 计算测试集中的Vector到训练集中所有Vector的距离。
+ 在训练集中，选取Cosine距离当前测试数据最近的k(k=5)个数据的label。
+ 根据距离给每个label的vote赋予不同的权重：$w=\frac{1}{d^2}$
+ 选择投票最高的label作为当前测试数据的label，从而实现分类。
+ 利用sklearn.neighbor的BallTree提高算法效率。


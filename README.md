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

#### 实验步骤

##### VSM模型

+ 使用textblob package进行数据预处理。包括：Normalization、Tokenization、Stemming、Stopwords。
+ 建立每个文件的字典，计算Term Frequency。
+ 均匀从每个类中采样，选取80%训练集，20%测试集合。
+ 根据训练集字典建立Global dictionary同时计算Document Frequency。
+ 利用TF-IDF方法计算每个Term的Weight，并据此建立Vector。

##### KNN模型

+ 计算测试集中的Vector到训练集中所有Vector的距离。
+ 在训练集中，选取Cosine距离当前测试数据最近的k(k=5)个数据的label。
+ 根据距离给每个label的vote赋予不同的权重：w = 1 / d^2
+ 选择投票最高的label作为当前测试数据的label，从而实现分类。
+ 利用sklearn.neighbor的BallTree提高算法效率。



### Homework2

#### 运行代码

~~~python
python -u NBC.py > result.log
~~~

#### 实验内容

实现朴素贝叶斯分类器，测试其在20 Newsgroups数据集上的效果。

#### 实验环境

- Windows 8.1
- Python 3.6(Anaconda)
- CPU-only

#### 实验步骤

__步骤1:__遍历training_path，计算所有文件的个数total_count，字典 class_file_count保存每一类文件的个数，以及字符串列表保存所有类的名字class_name。

__步骤2:__将global dict中的每个单词映射到下标；将class name中的每一个类名映射到下标。

__步骤3:__遍历training_path对应文件的字典，根据步骤2生成的映射关系，建立一个大小为(class_num, term_num)的ref_matrix保存每一类中每个单词出现的频数

__步骤4:__遍历testing_path，对testing_path的每一个数据，根据公式

P(y)=类别为y的文件个数 / 所有文件的个数

P(x_j|y)=(类别为y的文件中词语x_j出现的次数+1) / (类别为y的文件中所有词语的次数+被统计的词表中词语个数)

分别计算该数据属于每个类别的概率。



### Homework3

#### 运行代码

~~~python
python sklearn_cluster.py > res.log
~~~

#### 实验内容

测试sklearn中以下聚类算法在tweets数据集上的聚类效果。k-means, AffinityPropagation, mean-shift, SpectralClustering, AgglomerativeClustering, DBSCAN, GaussianMixtures。

#### 实验环境

- ubuntu16.04
- python3.7
- sklearn 0.19.2

#### 实验步骤

- sklearn.cluster.KMeans  => n_clusters 聚类个数  random_state 随机初始化
- sklearn.cluster.AffinityPropagation ==> damping 阻尼系数，防止抖动
- sklearn.cluster.MeanShift ==> bandwidth 高斯核的参数(类似于决定高纬球的半径) bin_seeding 优化加速
- sklearn.cluster.SpectralClustering ==> n_clusters 聚类个数 random_state 随机初始化
- sklearn.cluster.AgglomerativeClustering ==> n_clusters 聚类个数 linkage 确定计算距离的方法(ward, average, complete)
- sklearn.cluster.DBSCAN ==> eps 确定邻域的大小 min_samples 密集区域所需最小样本点
- sklearn.mixture.GaussianMixture ==> n_components 聚类个数
- sklearn.cluster.Birch ==> n_clusters 聚类个数 threshold 高维球半径 branching_factor 分支个数

通过sklearn.metrics.cluster的normalized_mutual_info_score方法，计算得分，判断聚类好坏。
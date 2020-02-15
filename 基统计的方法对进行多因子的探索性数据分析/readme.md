**在建立机器学模型之前，我们常常会对我们所拥有的特征进行探索性因子分析，探索性因子分析可以分为单因子分析和多因子分析。单因子分析主要针对某一个特征进行分析，分析方法往往采用统计指标（均值，中位数，众数，偏度系数和峰度系数等）以及图形可视化分析；而多因子分析主要是针对两个或两个以上的特征做联合分析，分析方法有检验分析（如：T检验分析，方差分析，卡方检验分析）、相关性分析、主成分分析、因子分析等，本文主要是记录一些多因子分析方法.**


## 1、假设检验
!![在这里插入图片描述](https://img-blog.csdnimg.cn/20200213102152184.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212194306805.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
```py
## 这是一个检验变量是否呈现正态分布的方法，基于偏度和峰度的检验方法。
import pandas as pd
import numpy as np
from scipy import stats
pts = 1000
np.random.seed(28041990)
a = np.random.normal(0, 1, size=pts) ##生成一个均值为0，标准差为1的1000个正太分布随机数
b = np.random.normal(2, 1, size=pts) ##生成一个均值为2，标准差为1的1000个正太分布随机数 
x = np.concatenate((a, b)) ##合并这两个数组
k2, p = stats.normaltest(x) ##k2表示统计量的值，p为p值
alpha = 1e-3 ##阀值
print("p = {:g}".format(p))
p = 3.27207e-11
if p < alpha: # null hypothesis: x comes from a normal distribution
	print("The null hypothesis can be rejected")
else:
	print("The null hypothesis cannot be rejected")


```
### 1.1 t检验
==主要是用来检验两组分布是否具有一致性==
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200213205909322.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
```py
import pandas as pd
import numpy as np
from scipy import stats as ss
ss.ttest_ind(ss.norm.rvs(size=10), ss.norm.rvs(size=20))
##out:Ttest_indResult(statistic=1.9250976356002707, pvalue=0.06443061130874687)
ss.ttest_ind(ss.norm.rvs(size=10), ss.norm.rvs(loc=1,scale=0.1,size=20))
## out:Ttest_indResult(statistic=-3.3034115592617534, pvalue=0.002617523871754732)
```
### 1.2 卡方检验
==卡方检验，用称之为四格检验方法，主要是用来检验两个因素是否具有比较强的联系==，如下：我们看一下性别与化妆与否是否具有关系，
H0：性别与化妆与否之间没有关系
H1：性别与化妆与否之间具有关系



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212195650929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212195829479.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
卡方直129.3是大于显著性水平为0.05的卡方值3.841，所以应该拒绝原假设，接受备择假设，即性别与男女化妆与否之间有比较强的关系。

```PY
import pandas as pd
import numpy as np
from scipy import stats
k2,p,_,_ss.chi2_contingency([[15, 95], [85, 5]], False
out:k2=129.29292929292927,p=5.8513140262808924e-30
```
### 1.3方差检验
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212203310211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
SST：总平方和或总变差平方和
SSM：组间平方和或平均平方平方和
SSE ：组内平方和或残差平方和
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212203517454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212203153503.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
F0:三种电池之间的平均寿命无差异
F1:三种电池之间平均寿命没有差异


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200212203642641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)

p值小于显著性水平，拒绝原假设，即认为三种电池的平均寿命具有差异性。


```py
from scipy import stats as ss
ss.f_oneway([49, 50, 39,40,43], [28, 32, 30,26,34], [38,40,45,42,48])
```

### 1.4 qq图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200213211942816.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
```py
from statsmodels.graphics.api import qqplot
from matplotlib import pyplot as plt
qqplot(ss.norm.rvs(size=100))#QQ图
plt.show()

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200213215401363.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
理论分位数值与样本分布正太分位数值在对角线上
## 2 相关系数
### 2.1 Pearson相关系数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200213102624895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
```py
s = pd.Series([0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5])
df = pd.DataFrame([[0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5], [0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1]])
#相关分析
print(s.corr(pd.Series([0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1])))
print(df.corr())
```



### 2.2 Spearman相关系数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200213102927639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
```py
import pandas as pd
df = pd.DataFrame([[0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5], [0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1]])
df.corr(method="spearman")
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200213220442407.png)
## 3 复合分析
### 3.1 交叉分析
（1）检验的方法，这里主要是利用HR_data.csv数据，观察部门之间员工离职率是否具有差异性。
```py
##看部门两两之间离职率是否具有差异性，用t检验的方法。
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
sns.set_context(context="poster",font_scale=1.2)
import matplotlib.pyplot as plt
df=pd.read_csv("./data/HR_data.csv")
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214170438890.png)
```py
dp_indices=df.groupby(by="department").indices
sales_values=df["left"].iloc[dp_indices["sales"]].values
technical_values=df["left"].iloc[dp_indices["technical"]].values
print(ss.ttest_ind(sales_values,technical_values))
dp_keys=list(dp_indices.keys())
dp_t_mat=np.zeros((len(dp_keys),len(dp_keys)))

for i in range(len(dp_keys)):
   for j in range(len(dp_keys)):
         p_value=ss.ttest_ind(df["left"].iloc[dp_indices[dp_keys[i]]].values,\
                                     df["left"].iloc[dp_indices[dp_keys[j]]].values)[1]
         if p_value<0.05:
             dp_t_mat[i][j]=-1 ## 拒绝原假设，认为两个部门离职率有差异性
         else:
             dp_t_mat[i][j]=p_value  ##接受原假设
sns.heatmap(dp_t_mat,xticklabels=dp_keys,yticklabels=dp_keys)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214193041222.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
如上图中黑色方框表示两个部门的离职具有差异性。
（2）透视表的方法
```py
piv_tb=pd.pivot_table(df, values="left", index=["department", "salary"], columns=["time_spend_company"],aggfunc=np.mean)
#piv_tb=pd.pivot_table(df, values="left", index=["department", "salary"], columns=["time_spend_company"],aggfunc=np.sum)
#piv_tb=pd.pivot_table(df, values="left", index=["department", "salary"], columns=["time_spend_company"],aggfunc=len)

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214200842141.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214200919800.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214200208938.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)




### 3.2 分组分析
(1) 离散值
```py
sns.barplot(x="salary",y="left",hue="department",data=df)
plt.show()   #按照部门分组（图例）hue参数， salary为x轴

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020021420232275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)(2) 连续值先分组，在做聚合

根据
	拐点（二街查分）、
	聚类、
	基尼系数
把连续值分类
```py
sl_s=df["satisfaction_level"]
sns.barplot(range(len(sl_s)),sl_s.sort_values())
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214203947633.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)

### 3.3 因子分析
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214212418739.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
（1）探索性因子分析
通过协方差矩阵，分析多元变量的本质结构，并可以转化、降维操作，得到空间中影响目标属性的最主要因子，例如主成分分析方法。
```py
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
sns.set_context(context="poster",font_scale=1.2)
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
df=pd.read_csv("./data/HR.csv")
#相关图
sns.heatmap(df.corr())
sns.heatmap(df.corr(), vmax=1, vmin=-1)
plt.show()
#PCA降维
my_pca=PCA(n_components=7)
lower_mat=my_pca.fit_transform(df.drop(labels=["salary","department","left"],axis=1).values)
print(my_pca.explained_variance_ratio_)
#sns.heatmap(pd.DataFrame(lower_mat).corr())
#plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200214214048312.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMjAzNQ==,size_16,color_FFFFFF,t_70)
降维后的矩阵使得各个变量之间都是正交的，及相关系数为1.

（2）验证性因子分析
测试一个因子与相对应的测度项之间的关系是否符合研究者所设计的理论关系。


# 线性分类

从线性回归出发，线性回归的三个特点分别是：**线性**、**全局性**、**数据未加工**

打破其中的一个或几个特点从而形成了其他的机器学习模型。

- 比如打破了**属性线性**的特点即**属性非线性**，就可以变成**特征转换（多项式回归）**
- 比如打破**全局线性**的特点即**全局非线性**，就变成了**线性分类（激活函数是非线性）**

![](imge/LinearClassification_1.png)


线性分类，可以看成**线性回归通过激活函数**带来了分类的效果，也可以看作是把数据**降维**然后根据阈值判断类别。

$$
y=f(w^Tx+b),\ y\in = \begin{cases}
  \{0,1\} \\
  [0,1]
\end{cases}
$$

函数$f$就是激活函数$activation\ function$

线性分类也有两种分类，一种是硬分类，一种是软分类

$$
线性分类 \begin{cases}
  硬分类 \ y\in \{0, 1\}  \begin{cases}
    线性判别式 \\
    感知机
  \end{cases} \\
  软分类 \ y\in [0,1] \begin{cases}
    生成式：Gaussian Discriminant Analysis\\
    判别式：Logistic Regression
  \end{cases}
\end{cases}
$$

## 几个模型

### [1. 感知机](./Perceptron.md)

### [2. 线性判别分析（Linear Discriminant Analysis）](LinearDiscriminantAnalysis.md)

### [3. 逻辑回归（Logistics Regression）](LogisticRegression.md)

### [4. 高斯判别分析（Gaussian Discriminant Analysis）](GaussianDiscriminantAnalysis.md)

### [5.朴素贝叶斯](NaiveBayes.md)



## 参考资料

- [机器学习-白板推导系列(四)-线性分类（Linear Classification）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV15W41127L2?spm_id_from=333.999.0.0)
- [逻辑回归（logistic regression）原理详解_guoziqing506的博客-CSDN博客_逻辑回归原理](https://blog.csdn.net/guoziqing506/article/details/81328402?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163888824516780255271123%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163888824516780255271123&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-3-81328402.pc_search_result_cache&utm_term=%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92&spm=1018.2226.3001.4187)


# 逻辑回归

从线性回归出发，线性回归的三个特点分别是：**线性**、**全局性**、**数据未加工**

打破其中的一个或几个特点从而形成了其他的机器学习模型。

比如打破了**属性线性**的特点即**属性非线性**，就可以变成**特征转换（多项式回归）**

比如打破**全局线性**的特点即**全局非线性**，就变成了**线性分类（激活函数是非线性）**

![](imge/LogisticRegression_1.png)



## 1. 线性分类

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

## 2. 
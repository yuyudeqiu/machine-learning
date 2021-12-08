## 4. 高斯判别分析（Gaussian Discriminant Analysis）

高斯判别分析属于是**概率生成模型**，概率判别模型是主要是求出y=0的概率和y=1的概率$\hat{y}=\argmax_{y\in \{ 0,1 \}} P(y|x)$。

生成模型不是直接求出$P(Y|X)$而关心的是$P(y=0|x)$和$P(y=1|x)$两者谁大，并不是求一个确切的大小，所以可以通过贝叶斯公式$P(y|x)=\frac{P(x|y)P(y)}{P(x)}$去比较大小

而其中的$P(x)$并与y实际上是没有关系的，所以可以得到$P(y|x)\propto P(x|y)P(y)$，而$P(y|x)\propto P(x|y)P(y)$实际上就是联合概率$P(x,y)$，所以主要就是对联合概率进行建模。

所以$P(y)$可以认为是先验prior, $P(y|x)$可以认为是似然likelihood，$P(y|x)$就是后验posterior

也就是让后验最大化

$$
\hat{y} = \argmax_{y\in \{ 0,1 \}} P(y|x) = \argmax_y P(y)P(x|y)
$$

### 4.1 模型定义

高斯判别分析主要是假设$y$满足伯努利分布$y \sim Bernoulli (\Phi)$，很好理解，因为y不是0就是1。假设$x|y$满足高斯分布 $x|y=1 \sim N(\mu_1, \Sigma)$，$x|y=0 \sim N(\mu_2, \Sigma)$

直接写出参数$\theta$的对数似然函数

$$
\begin{aligned}
log-likelihood:l(\theta) &= \log \prod_{i=1}^{N}P(x_i,y_i) \\
&= \sum_{i=1}^{N} \log (P(x_i|y_i P(y_i))) \\
&= \sum_{i=1}^{N} [\log P(x_i|y_i) + \log P(y_i)] \\
&= \sum_{i=1}^{N} [\log N(\mu_1,\Sigma)^{y_i}N(\mu_2,\Sigma)^{1-y_i} + \log \Phi^{y_i}(1-\Phi^{1-y_i})] \\
&= \sum_{i=1}^{N} [\log N(\mu_1,\Sigma)^{y_i} +\log N(\mu_2,\Sigma)^{1-y_i} + \log \Phi^{y_i}(1-\Phi^{1-y_i})] 
\end{aligned}
$$

这里$\theta$可以写成一个元组的形式$\theta=(\mu_1,\mu_2,\Sigma,\Phi)$

求解目标就是$\hat{\theta}=\argmax_\theta l(\theta)$

### 4.2 模型求解

主要就是求解$\theta=(\mu_1,\mu_2,\Sigma,\Phi)$里面是四个参数。

上面已经将对数似然函数推导到三项，$\sum_{i=1}^{N} [\log N(\mu_1,\Sigma)^{y_i} +\log N(\mu_2,\Sigma)^{1-y_i} + \log \Phi^{y_i}(1-\Phi^{1-y_i})]$

从最简单的一项开始，也就是第三项，求$\Phi$：

$$
\begin{aligned}
  ③ &= \sum_{i=1}^{N} [y_i \log \Phi + (1-y_i)\log(1-\Phi)] \\
  \frac{\partial ③}{\partial \Phi} &= \sum_{i=1}^{N} [y_i \frac{1}{\Phi} - (1-y_i)\frac{1}{1-\Phi}] =0 \\
\end{aligned} \\

\begin{aligned}
  &\Rightarrow \sum_{i=1}^{N} [y_i (1-\Phi) - (1-y_i) \Phi] = 0 \\
  &\Rightarrow \sum_{i=1}^{N} (y_i-\Phi) = 0 \\
  &\Rightarrow y_i -N \Phi = 0 \\
  &\Rightarrow \hat{\Phi} = \frac{1}{N} \sum_{i=1}^{N}y_i = \frac{N_1}{N}
\end{aligned}
$$

然后是$\mu_1$和$\mu_2$，两个的求解过程是等价的，求解一个就可以同理得到另一个，求解$\mu_1$：

$$
  \begin{aligned}
    ① &= \sum_{i=1}^{N} \log N(\mu_1, \Sigma)^{y_i}  \\
      &= \sum_{i=1}^{N} y_i \log \frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}} exp(-\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1))
  \end{aligned}
$$

这里因为是求解$\mu_1$，所以可以省去有$\Sigma$的一项，即：

$$
\mu_1 = \argmax_{\mu_1} ① = \argmax_{\mu_1} \sum_{i=1}^{N} y_i(- \frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1))
$$

$$
\begin{aligned}
\Delta &= \sum_{i=1}^{N} y_i(- \frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)) \\
       &= - \frac{1}{2} \sum_{i=1}^{N} y_i (x_i^T \Sigma^{-1} - \mu_1 \Sigma^{-1} )(x_i-\mu_1) \\
       &= - \frac{1}{2} \sum_{i=1}^{N} y_i (x_i^T \Sigma^{-1} x_i - 2 \mu_1 \Sigma^{-1} x_i + \mu_1 \Sigma^{-1} \mu_1)
\end{aligned}
$$

求导：

上面$\Delta$的第一项实际上是一个常数，所以求导过程中可以直接被消去
$$
\begin{aligned}
  &\Rightarrow \frac{\partial \Delta}{\partial \mu_1} = \sum_{i=1}^{N} y_i(\Sigma^{-1}x_i + \Sigma^{-1}\mu_1) = 0 \\
  &\Rightarrow \sum_{i=1}^{N} y_i(\mu_1 - x_i) = 0 \\
  &\Rightarrow \sum_{i=1}^{N} y_i \mu_1 = \sum_{i=1}^{N} y_i x_i \\
  &\Rightarrow \hat{\mu_1} = \frac{\sum_{i=1}^{N} y_i x_i}{\sum_{i=1}^{N} y_i} = \frac{\sum_{i=1}^{N} y_i x_i}{N_1} 
\end{aligned}
$$

同理可得：
$$
\hat{\mu_2} =  \frac{\sum_{i=1}^{N} y_i x_i}{N_2}
$$

然后就是求$\Sigma$

![](imge/LinearClassification_5.png)

这里不是很能理解直接得到结论吧。。。。

$$
\hat{\Sigma} = \frac{1}{N}(N_1S_1+ N_2S_2)
$$

## 2. 线性判别分析（Linear Discriminant Analysis）

### 2.1 思想

先对数据符号做一些假设（设定）

X是数据特征集合，$N\times p$的矩阵

$$
X=(x_1,x_2,...,x_N)^T = \begin{pmatrix}
  x_1^T\\
  x_2^T\\
  ...\\
  x_N^T
\end{pmatrix}
$$

Y是标签集合，$N\times 1$的矩阵

$$
Y=\begin{pmatrix}
  y_1\\
  y_2\\
  ...\\
  y_N
\end{pmatrix}
$$

数据集可以表示为：
$$
\{ (x_i,y_i) \}_{i=1}^{N}\\
x_i是p维的，y_i\in \{ +1, -1 \}
$$

线性判别分析的思想可以总结为：**类内小，类间大**

**实际上可以理解为一个降维再分类的一个过程**。

![](imge/LinearClassification_3.png)

比如上图，可以通过将坐标轴XY上的点映射到红色的坐标轴上变成一维，再选取适合的阈值进行分类。优化目标自然是找到适合投影的一个方向。而从图中，阈值的点，垂直于映射平面的超平面就是所求的分类的超平面，如图蓝色的线。

上面所说的类内小，实际上就是让类内的方差尽量的小，也就是同一类的映射之后越紧凑越好。想法类间大就是不同类之间月松散越好。

### 2.2 loss function 的构建

假设点映射到超平面的值是$z_i=w^Tx_i$

则
$$
\overline{z}=\frac{1}{N}\sum_{i=1}^{N}z_i=\frac{1}{N} \sum_{i=1}^{N} w^Tx_i
$$

$z$的方差可以表示为$S_z$

$$
\begin{aligned}
S_z &= \frac{1}{N}\sum_{i=1}^{N}(z_i-\overline{z})(z_i-\overline{z})^T \\
&= \frac{1}{N}\sum_{i=1}^{N}(w^Tx_i-\overline{z})(w^Tx_i-\overline{z})^T
\end{aligned}
$$

可以把X根据Y的不同分为两个集合，$x_{c1}=\{ x_i|y_i=+1 \}$,$x_{c2}=\{ x_i|y_i=-1 \}$

$|x_{c1}|=N_1$,\ $|x_{c2}|=N_2$, $N_1+N_2=N$

所以可以分别计算$c_1$和$c_2$的均值以及方差$\overline{x_{c_1}}$，$\overline{x_{c_2}}$，$S_{c_1}$，$S_{c_2}$

$$
\begin{aligned}
  c_1:\overline{z_1} &= \frac{1}{N_1}\sum_{i=1}^{N_1}w^Tx_i \\
  S_1 &= \frac{1}{N_1}\sum_{i=1}^{N_1}(w^Tx_i-\overline{z_1})(w^Tx_i-\overline{z_1})^T \\
  c_2:\overline{z_2} &= \frac{1}{N_2}\sum_{i=1}^{N_2}w^Tx_i \\
  S_2 &= \frac{1}{N_2}\sum_{i=1}^{N_2}(w^Tx_i-\overline{z_2})(w^Tx_i-\overline{z_2})^T \\
\end{aligned}
$$

类间大，可以理解为是让$c_1$和$c_2$之间的均值差距大，用$(\overline{z_1}-\overline{z_2})^2$表示

类内小：可以表示为让每个$S$都小，用$S_1+S_2$表示

所以可以构建出目标函数就是

$$
J(w)=\frac{(\overline{z_1}-\overline{z_2})^2}{S_1+S_2} \\
$$

而优化目标就是
$$
\hat{w}=arg\max_w J(w)
$$

将$\overline{Z_1}$, $\overline{Z_2}$, $S_1$,$S_2$都代入$J(w)$进一步推导

$$
\begin{aligned}
J(w) &= \frac{(\overline{z_1}-\overline{z_2})^2}{S_1+S_2} \\
分子 &= ( \frac{1}{N_1}\sum_{i=1}^{N_1}w^Tx_i - \frac{1}{N_2}\sum_{i=1}^{N_2}w^Tx_i )^2 \\
&=[w^T(\frac{1}{N_1}\sum_{i=1}^{N_1}x_i - \frac{1}{N_2}\sum_{i=1}^{N_2}x_i)]^2 \\
&=[w^T(\overline{x_{c_1}} - \overline{x_{c_2}})]^2 \\
&=w^T(\overline{x_{c_1}} - \overline{x_{c_2}})(\overline{x_{c_1}} - \overline{x_{c_2}})^Tw\\
分母&=S_1+S_2 \\
S_1 &= \frac{1}{N_1}\sum_{i=1}^{N_1}(w^Tx_i-\overline{z_1})(w^Tx_i-\overline{z_1})^T \\
&= \frac{1}{N_1}\sum_{i=1}^{N_1}w^T(x_i-\overline{x_{c_1}})(x_i-\overline{x_{c_1}})^Tw \\
&= w^T [\frac{1}{N_1}\sum_{i=1}^{N_1}(x_i-\overline{x_{c_1}})(x_i-\overline{x_{c_1}})^T]w \\
&= w^T S_{c_1}w\\
分母 &=w^T S_{c_1}w+w^T S_{c_2}w \\
&= w^T (S_{c_1}+S_{c_2})w
\end{aligned}
$$

可以得到：

$$
\begin{aligned}
J(w) &=\frac{(\overline{z_1}-\overline{z_2})^2}{S_1+S_2}\\
&= \frac{w^T(\overline{x_{c_1}} - \overline{x_{c_2}})(\overline{x_{c_1}} - \overline{x_{c_2}})^Tw}{w^T (S_{c_1}+S_{c_2})w}  
\end{aligned}
$$

### 2.3 目标函数求解

得到目标函数之后，就可开始对模型进行求解了。

$$
\begin{aligned}
J(w) &=\frac{(\overline{z_1}-\overline{z_2})^2}{S_1+S_2}\\
&= \frac{w^T(\overline{x_{c_1}} - \overline{x_{c_2}})(\overline{x_{c_1}} - \overline{x_{c_2}})^Tw}{w^T (S_{c_1}+S_{c_2})w}  \\
&=\frac{w^TS_bw}{w^TS_ww}
\end{aligned} 
$$

$S_b$：between-class 内类方差

$S_w$：with-class 类内方差

对$J(w)$求导：

$$
\begin{aligned}
  J(w) &= \frac{w^TS_bw}{w^TS_ww} \\
  &= w^TS_bw(w^TS_ww)^{-1}
\end{aligned}
$$

$$
\begin{aligned}
  \frac{\partial{J(w)}}{\partial{w}} &= S_bw(w^TS_ww)^{-1} - w^TS_bw(w^TS_ww)^{-2}S_ww=0 \\  
\end{aligned}
$$

可得：

$$
\begin{aligned}
  w^TS_bwS_ww &=S_bw(w^TS_ww) \\
\end{aligned}
$$

到这一步，可以发现$w^TS_bw$和$(w^TS_ww)$实际上都是实数，所以：

$$
\begin{aligned}
  S_ww &= \frac{w^TS_ww}{w^TS_bw}S_bw \\
  w &= \frac{w^TS_ww}{w^TS_bw}S_w^{-1}S_bw
\end{aligned}
$$

由于这个求解的$w$并不在意它的大小，而是需要求解的是他的一个方向，所以可以忽略实数部分对w方向的影响。进而得到：

$$
\begin{aligned}
w &\propto S_w^{-1}S_bw \\
  &\propto (\overline{x_{c_1}} - \overline{x_{c_2}})(\overline{x_{c_1}} - \overline{x_{c_2}})^Tw \\
  &\propto S_w^{-1}(\overline{x_{c_1}} - \overline{x_{c_2}})
\end{aligned}
$$

从第二行到第三行的变换是同理之上的 因为$(\overline{x_{c_1}} - \overline{x_{c_2}})^Tw$的结果也是一个实数，对求解w的方向来说无影响，直接丢掉。

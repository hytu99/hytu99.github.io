---
title: 【机器学习】3. Naive Bayes
top: false
cover: false
toc: true
mathjax: true
date: 2019-10-26 17:15:48
password:
summary:
tags:
- 机器学习
- 学习笔记
categories: 机器学习
---

> 整理自同学的笔记。
> 监督学习是指有目标变量或预测目标的机器学习方法，包括分类和回归。

###  分类中的朴素贝叶斯方法（Naive Bayes Classifier)

以垃圾邮件的分类（Span Detecor）为例。

#### 目标

对于训练过的模型，给定${\bf x}$，给出$P(spam|{\bf x})$。

训练数据记作$ \\{ \mathbf{x} _ i,   y _ i \\} $， $y_i \in {\mathcal C} = \\{spam, not \\ _ spam \\} $。

#### 基本假设

1. 属性值$x_i$条件独立于标签值，即
   $$
   P(x_1, x_2, \cdots ,x_{| \cal{X} |}|{\mathcal C}) = \prod_i P(x_i | {\mathcal C })
   $$
   
   以垃圾邮件分类为例，该问题中的样本${\bf x}_i$ 为表征邮件属性的矢量（比如词向量），表示邮件的整体特征。如果不考虑这一假设，在通常的采样中对$P({\bf x}|c)$的估计往往会导出很小的值（不容易找到两封一样的邮件）。

   而这一假设为我们带来的好处则是摆脱了属性捆绑的桎梏，将单个属性作为统计与概率估计的原子单位，既提高了对数据的利用率也有效地降低了模型需要的参数数目。当然这以真实性为代价。
   
2. 属性值的分布独立于其出现的位置：

   $$
   P(x_i = w_k|c) = P(x_j=w_k|c),\forall i\not=j
   $$

   亦即：

   $$
   P(x_i = w_k|c) = P(w_k|c),\forall i
   $$

   这一条件是我们脱离了对邮件长度与位置的依赖，估计中我们就只需要考虑词频，进一步降低了估计参数的数目和复杂度。
   
#### 理论依据（贝叶斯定理）

$$
\begin{align}
{\hat y} &= \arg \max _ {c \in {\mathcal C} } P(c | {\bf x} ) \\\\
&= \arg \max _ {c \in {\mathcal C} } \frac{P({\bf x}|c) P(c) } {P({\bf x} ) } \\\\
&= \arg \max _ {c \in {\mathcal C} } P({\bf x}|c)P(c) \\\\
&= \arg \max _ {c \in {\mathcal C} } P(c) \prod_i P(x_i | c)  \ (assumption \ 1)  \\\\ 
&= \arg \max _ {c \in {\mathcal C} } P(c) \prod_k P(w_k|c) ^ {t _ k} \ (assumption \ 2) \\\\
\end{align}
$$

其中的$P(c)$为先验概率，从采样数据中估计。使先验概率更接近真实分布这一点对采样的多样性提出了一定的要求。

最后的$P(w _ k|c)$可以用表示$P(w_k|c) = \dfrac{n_{ck}}{n_c}$,其中$n_c=\sum_{i : y=c} | x _ i| $表示c类出现的次数，$n_{ck}$表示c类中词$w_k$出现的次数。但是注意到如果在采样中只要有$n_{c k}=0$,那在估计中就一定会有$P(w_k|c)=0$,这在实际中并不是合理的。为了解决这种问题，有一种方案是Laplace Smoothing:

$$
P(w_k|c) = \frac{n_{c k} + 1} {n _ c + | \mathcal{V} | }
$$

#### 朴素贝叶斯分类器训练（Training Naive Bayes Classifier）

$$
\begin{align}
&\text{Input: trainning samples } {\mathcal D} = \\{ ({\bf x_i},y_i) \\} \\\\
& {\mathcal V} \leftarrow \text{the set of distinct words and other tokens in }  {\mathcal D} \\\\
& \text{for each target value } c \in {\mathcal C}, \text{ do} \\\\
& ~ ~ ~ ~ ~ ~ ~ ~  {\mathcal D_c} \leftarrow \text{the training samples whose labels are c} \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ P(c) \leftarrow \dfrac{|{\mathcal D_c}|}{|{\mathcal D}|} \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ T_c \leftarrow \text{a single document by concentrating all training samples in } \mathcal{D} _ c \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ n_c \leftarrow |T_c| \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ \text{for } w_k \in \cal{V} \text{ do} \\\\
&  ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ n_{ck} \leftarrow  \text{the number of times the word } w_k  \text{ occurs in } T_c \\\\
&  ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ P(w_k|c) = \dfrac{n _ {ck} + 1}{n_c+ | \mathcal{V} | } \\\\
&  ~ ~ ~ ~ ~ ~ ~ ~ \text{endfor} \\\\
& \text{endfor}
\end{align}
$$

所谓训练，就是计算$P(w_k|c)$的表罢了。

#### 朴素贝叶斯分类器测试（Testing Naive Bayes Classifier）

$$
\begin{align}
& \text{Input: A new sample } {\bf x}, \text{ 设} x_i \text{是} {\bf x}  \text{的第 i 个属性}, I = \emptyset \\\\
& \text{for } x_1, \cdots, x_i \text{ do} \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ \text{if } \exists w_k \in \mathcal{V} \text{ such that } w_k = x_i, \text{ then} \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ I \leftarrow I \cup k \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ \text{end if} \\\\
& \text{end for} \\\\
& \text{predict the label of } \mathbf{x} \text{ by } \hat y= \arg \max_{c \in {\mathcal C}}P(c) \prod _ { i \in I} P(w_i|c) 
\end{align}
$$

这个算法虽然简单，但是好用。

#### 算法性能的衡量指标

1. 准确率（Accuracy)
   $$
   \text{Accuracy} = \frac{\text{correctly predicted samples} } {\text{total number of samples} }
   $$
   这个指标并不适用于一般情景，它忽略了两种分类错误的不同风险。

2. 查准率（Precision)、召回率（Recall）、F-score：

   |           | T（正确）    | F（错误）                 | 总计     |
   | --------- | ------------ | ------------------------- | -------- |
   | P（正例） | TP           | FP（第一类错误， 假正例） | 正例总数 |
   | N（反例） | TN           | FN（第二类错误， 假反例） | 反例总数 |
   | 总计      | 预测正确总数 | 预测错误总数              | 样例总数 |

   则
   $$
   \begin{align}
   & \text{Precision} = \frac{TP}{TP+FP} \\\\
   & \text{Recall} = \frac{TP}{TP+FN}  \\\\
   & F_1 = \dfrac{2}{\dfrac{1}{\text{Precision} } + \dfrac{1}{\text{Recall} } }
   \end{align}
   $$
   

### 逻辑斯谛回归（Logistic Regression)

#### 目标：

给定集合$ \\{ ({\bf x}_i, y_i \\} ^ n _ {i=1} $, 其中$y_i \in \{0,1\}$,寻找映射：
$$
f:X\rightarrow Y, where\ X=(X_1,\cdots,X_d)\ and\ Y\in\{0,1\}
$$

#### 基本假设：

1. $Y \sim Bern(P)$, $Y$ 服从伯努利二项分布，$P(Y=1) = p$.
2. $X = (X_1,\cdots,X_d)$中的$X_j$是连续随机变量。
3. 高斯分布: $P(X_j|Y=0)\sim N(\mu _ {j0,\sigma _ j^2}),P(X_j|Y=1)\sim  N(\mu _ {j1,\sigma_j^2})$
4. $X_i, X_j$条件独立于$Y$, $\forall i\not=j$.

#### 理论依据：

综上,

$$
\begin{align}
P(Y=0|X) &= \dfrac{P(X|Y=0)P(Y=0)}{P(X|Y=0)P(Y=0)+P(X|Y=1)P(Y=1) }
\\\\ &= \dfrac{1}{1+\dfrac{P(X|Y=1)P(Y=1)}{P(X|Y=0)P(Y=0)} }
\\\\ &= \dfrac{1}{1+\exp (\ln (\dfrac{P(X|Y=1)P(Y=1)}{P(X|Y=0)P(Y=0) } ) ) }
\\\\ &= \dfrac{1}{1+ \exp(\sum_j \ln (\dfrac{P(X_j|Y=1)}{P(X_j|Y=0)})+ \ln \dfrac{p}{1-p})} (assumption\ 4)
\end{align}
$$

而

$$
\begin{align}
\sum_j \ln (\frac{P(X_j|Y=1)}{P(X_j|Y=0)}) &= \sum_j \ln (\frac{\exp (-\dfrac{(X_j-\mu_{j1}) ^ 2}{2\sigma_j ^ 2})}{\exp (-\dfrac{(X_j-\mu_{j0}) ^ 2}{2 \sigma_j^2})})(assumption\ 3)
\\\\ &= \sum_j\dfrac{\mu_{j1}-\mu_{j0}}{\sigma_j^2}X_j+\sum_j\frac{\mu_{j0}^2-\mu_{j1}^2}{2\sigma_j^2}
\end{align}
$$
将其带回原式，
$$
\begin{align}
P(Y=0|X) &= \frac{1}{1+ \exp (\sum_j \dfrac{\mu_{j1}-\mu_{j0} }{\sigma_ j^ 2} X_j + \sum_j \dfrac{\mu_{j0}^ 2- \mu_ {j1}^ 2} {2 \sigma_ j^ 2}+ \ln \dfrac{p}{1-p})}
\\\\  &= \frac{1}{1+ \exp (\sum_j w_j X_j + w_0)}
\end{align}
$$
于是又有
$$
\begin{align}
P(Y=1|X) &= \frac{\exp(\sum_jw_jX_j+w_0)}{1+ \exp(\sum_jw_jX_j+w_0)}
\end{align}
$$
可见决策平面$\sum_jw_jX_j+w_0=0$是线性的。当找到决策平面时，该分类问题就会迎刃而解。而下一步，我们就需要找出需要的权向量${\bf w}$。

**采用最大似然估计法：**
$$
\begin{align}
\hat {\bf w} &= \arg \max_ \mathbf{w} \prod_i P(y_i|X_i,{\bf w})
\\\\ &= \arg \max_ \mathbf{w} \sum_i \ln (P(y_i|X_i, {\bf w}) )
\end{align}
$$

令 $-L({\bf w}) = \sum_i(y_i \ln(P(Y=1|X_i,{\bf w}))+(1-y_i) \ln(P(Y=0|X_i,{\bf w})))$,则问题转化为：
$$
\hat{\bf w} = \arg \min_{\bf w}L({\bf w})
$$
那么似乎可以用梯度下降法来求解该问题。（解的存在性、唯一性（严格凸、强凸））

采用正则化可以保证这两点：
$$
\hat {\bf w} = \arg \max_{\bf w} L({\bf w})+\frac{\lambda}{2}\Vert{\bf w}\Vert_2^2
$$

对于多分类问题，可以训练多个分类器。其中$Y\in \cal{C} = \{c_1, \cdots, c_k\}$，可令

$$
\begin{align}
P(Y\not=c_k|X) &= \frac{1}{1+ \exp(\sum_jw_{k.j}X_j+w_0)}
\end{align}
$$

#### 实际问题：数据的不平衡性

来自不同分类的数据数目不平衡时，回导致训练得出的决策平面有更大的偏移。

解决方案包括：

- undersample（主要）

- oversample

---
title: 【机器学习笔记】1. Linear Regression
top: false
cover: false
toc: true
mathjax: true
date: 2019-10-23 17:11:01
password:
summary:
tags: 
- 机器学习
- 学习笔记
categories: 机器学习
---
> 整理自同学的笔记

### Linear Regression

Given a data set $ \\{ ( \mathbf{x}_i, y_i ) \\} _ { i = 1 } ^ n $, where  $\mathbf{x}_i \in \mathbb{R}^{d+1}$ and $y_i \in \mathbb{R}$.

#### Linear Regression by Least Squares
$$
y_i = \mathbf{w}^T \mathbf{x}_i,
$$

where $ \mathbf{x} _ i = (1, x_{ i1}, x_{i2}, \cdots , x_{id})^T $ and $ \mathbf{w} = (w_0, w_1, \cdots , w_d)^T $.

Average Fitting error is

$$
\begin{align}
L &= \frac{1}{n} \sum_i^n(y_i - \mathbf{w}^T\mathbf{x}_i)^2 \\\\
&= \frac{1}{n} \Vert \mathbf{y-\bar{X}w} \Vert ^2
\end{align}
$$

where $\mathbf{y}=(y_i, \cdots , y_n) $ and $\mathbf{\bar{X}} \in \mathbb{R}^{n \times (d+1)}$.

$$
0 = \left. \frac{\partial L}{\partial \mathbf{w}}\right|_\mathbf{w=\hat{w}}=-\frac{2}{n}\mathbf{\bar{X}}^T(\mathbf{y}-\mathbf{\bar{X}\hat{w}}) \Rightarrow \mathbf{\hat{w}}= (\mathbf{\bar{X}}^T\mathbf{\bar{X}})^{-1}\mathbf{\bar{X}}^T\mathbf{y}
$$

$$
\mathbf{\hat{y}} =\mathbf{\bar{X}}(\mathbf{\bar{X}}^T\mathbf{\bar{X}})^{-1}\mathbf{\bar{X}}^T\mathbf{y}
$$

Projection matrix $ P = \mathbf{\bar{X}}(\mathbf{\bar{X}}^T\mathbf{\bar{X}})^{-1}\mathbf{\bar{X}}^T$ projects an arbitrary vector to the column space of $\mathbf{\bar{X}}$.

#### Linear Regression by Maximum Likelihood
$$
y_i = \mathbf{w}^T\mathbf{x}_i + \epsilon_i , 
$$

假设$ \mathbf{w} $与$ \mathbf{x}_i$给定。

**Assumption 1**:  $\epsilon_i \sim \cal{N}(0, \sigma^2)$,  $y_i |\mathbf{x}_i, \mathbf{w}, \sigma \sim \cal{N}(\mathbf{w}^T\mathbf{x}_i, \sigma^2)$.

**Assumption 2**: IID. $P((\mathbf{x}_1, y_2),\cdots ,(\mathbf{x}_n, y_n)) = \prod_iP((\mathbf{x}_i, y_i))$.

$$
\begin{align}
L &= P(y_1, \cdots , y_n | \mathbf{x}_1, \cdots, \mathbf{x}_n, \mathbf{w}, \sigma) \\\\
 &= \frac{P((\mathbf{x}_1, y_2), \cdots ,(\mathbf{x}_n, y_n)|\mathbf{w}, \sigma)}{P(\mathbf{x} _ 1, \cdots ,\mathbf{x}_n|\mathbf{w}, \sigma)}\\\\
&=\frac{\prod _ {i=1} ^n P(\mathbf{x}_i, y_i|{\bf w}, \sigma)}{\prod _ {i=1}^n P(\mathbf{x}_i|{\bf w}, \sigma)} \\\\
 &= \prod _ {i=1}^n P(y_i|{\bf x},{\bf w}, \sigma)
\end{align}
$$

$$
\begin{align}
\log L &= \sum_i^n \log (\frac{1}{\sqrt{2 \pi \sigma^2}}\exp\{\frac{1}{2\sigma^2}(y_i-{\bf w}^T{\bf x}_i)^2\})\\\\
&=-\frac{n}{2}\log 2\pi - n \log \sigma - \frac{1}{2\sigma^2}\sum_i^n(y_i-{\bf w}^T{\bf x}_i)^2
\end{align}
$$

$$
\begin{align}
0 &=\left. \frac{\partial \log L}{\partial {\bf w} } \right|_ \mathbf{w=\hat{w}} \\\\
&=\frac{1}{\sigma^2}\sum_{i=1}^n{\bf x}_i y_i - {\bf x}_i{\bf x}_i^T{\bf \hat{w} } \\\\
&=\frac{1}{\sigma^2}(\mathbf{\bar{X}}^T\mathbf{y}-{\bf \bar{X} }^T\mathbf{\bar{X}\hat{w} })) \\\\
& \Rightarrow \mathbf{\hat{w} }= (\mathbf{\bar{X} }^T\mathbf{\bar{X} })^{-1}\mathbf{\bar{X} }^T\mathbf{y}
\end{align}
$$

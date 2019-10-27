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

#### 垃圾邮件的分类（Span Detecor）

##### 目标

对于训练过的模型，给定${\bf x}$，给出$P(spam|{\bf x})$。

训练数据记作$ \\{ \mathbf{x} _ i,   y _ i \\} $， $y_i \in {\mathcal C} = \\{spam, not \\ _ spam \\} $。

##### 基本假设

1. 属性值$x_i$条件独立于标签值，即
   $$
   P(x_1, x_2,...,x_{|x|}|{\mathcal C}) = \prod_iP(x_i|{\mathcal C})
   $$

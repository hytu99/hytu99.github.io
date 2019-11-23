---
title: 【机器学习笔记】2. Gradient Descent
top: false
cover: false
toc: true
mathjax: true
date: 2019-10-24 15:13:03
password:
summary:
tags:
- 机器学习
- 学习笔记
categories: 机器学习
---
> 整理自同学的笔记

### Gradient Descent

#### Part 1

$$
\min f(\mathbf{x})
$$

- $f(\mathbf{x})$ is continuously differentiable.

- $f(\mathbf{x})$ is convex.

-------
**Definition:** A set $C$ is convex if the line segment between any two points in $C$ lies in $C$. that is $\forall x_1,x_2 \in C$, and $\forall \theta \in [0, 1]$, we have $\theta x_1 + (1-\theta)x_2 \in C$.

**Definition:** A function $f: \mathbb{R}^n\rightarrow\mathbb{R}$ is convex if $\rm{dom}$ $f$ is convex and if $\mathbf{x},\mathbf{y} \in$ ${\rm dom}f$ and $\theta \in [0, 1]$, we have $f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})  $.

**Definition:** A function is strict convex if strict inequality holds where $\mathbf{x} \ne \mathbf{y}$ and  $\theta \in [0, 1]$.

**Definition:** A function is strongly convex with parameter $u$ if  $f - \dfrac{u}{2} \Vert \mathbf{x} \Vert ^2$ is convex.

---

**Theorem 1:** Suppose that $f$ is continuously differentiable. Then $f$ is convex if and only if $\rm{dom}$ $f$ is convex and $f(\mathbf{y}) \ge f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y}-\mathbf{x} \rangle, \forall \mathbf{x}, \mathbf{y} \in $ ${\rm dom}f$.

**Proof:** 

$( \Rightarrow )$ 

$$
\begin{align}
f(\mathbf{x} + \theta(\mathbf{y}-\mathbf{x})) &\le f(\mathbf{x}) + \theta [f(\mathbf{y}) - f(\mathbf{x})]\\\\
f(\mathbf{y}) - f(\mathbf{x}) &\ge \lim_{\theta \to 0} \frac{f(\mathbf{x} + \theta(\mathbf{y}-\mathbf{x}))}{\theta}\text{（方向导数）}\\\\
&= \langle \nabla f(\mathbf{x}), \mathbf{y}-\mathbf{x} \rangle
\end{align}
$$

$ ( \Leftarrow ) $

$$
\mathbf{z} = \theta \mathbf{x} + (1 - \theta) \mathbf{y}
$$

$$
 f(\mathbf{x}) \ge f(\mathbf{z}) + \langle \nabla f(\mathbf{z}), \mathbf{x} - \mathbf{z} \rangle \tag{1}
$$

$$
 f(\mathbf{y}) \ge f(\mathbf{z}) + \langle \nabla f(\mathbf{z}), \mathbf{y} - \mathbf{z} \rangle \tag{2}
$$

$ \theta (1) + (1 - \theta) (2)  $可得。

**Corollary:** Suppose $ f $ is continuously differentiable. Then $ f $ is convex iff  ${\rm dom}f$ is convex and $\langle \nabla f(\mathbf{x})- \nabla f(\mathbf{y}), \mathbf{x}  - \mathbf{y} \rangle \ge 0$.

**Theorem 2:** Suppose that $ f $ is continuously differentiable. Then $ f $ is convex iff ${\rm dom}f$  is convex and $ \nabla ^2 f (\mathbf{x}) \ge 0 $.

**Proof:** 

$ ( \Rightarrow ) $

Let $ \mathbf{x} _ t = \mathbf{x} + t \mathbf{s} $, $t > 0$. Then

$$
\begin{align}
0 \le \frac{1}{t^2} & \langle \nabla f(\mathbf{x} _ t) - \nabla f(\mathbf{x}) , \mathbf{x} _ t - \mathbf{x} \rangle \\\\
& = \frac {1}{t} \langle \nabla f(\mathbf{x} _ t) - \nabla f(\mathbf{x}), \mathbf{s} \rangle \\\\
& = \frac {1}{t} \int _ 0 ^ t \langle \nabla ^2 f(\mathbf{x} + \tau \mathbf{s})\mathbf{s}, \mathbf{s} \rangle d\tau \text{（微积分基本定理）}\\\\
& \xrightarrow{t \to 0} \langle \nabla ^2 f(\mathbf{x})\mathbf{s}, \mathbf{s} \rangle \\\\
& = \mathbf{s}^T \nabla ^2 f(\mathbf{x}) \mathbf{s}
\end{align}
$$

$ ( \Leftarrow ) $

$$
\begin{align}
g(t) & = f(\mathbf{x} + t \mathbf{s}) \\\\
g'(0) &= \langle \nabla f(\mathbf{x}), \mathbf{s} \rangle \\\\
g''(0) &= \langle \nabla ^2 f(\mathbf{x})\mathbf{s}, \mathbf{s} \rangle \\\\
g(1) & = g(0) + \int _ 0 ^ 1 g'(t) dt \\\\
&  = g(0) + \int _ 0 ^ 1 [g'(0) + \int _ 0 ^ t g''(\tau) d\tau ] dt \\\\
& \ge g(0) + g'(0) \\\\
f(\mathbf{x} +  \mathbf{s}) & \ge f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{s} \rangle
\end{align}
$$

**Theorem 3:** Suppose $f$ is  continuously differentiable. Then $ \mathbf{x}^  * \in \arg \min \limits _ \mathbf{x}  f ( \mathbf{x})  $  iff $ \nabla f(\mathbf{x}^ *) = 0$, $  f(\mathbf{y}) \ge f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle = f(\mathbf{x})$.



#### Part 2

$$
\min f(\mathbf{x})
$$

- The objective function $f(\mathbf{x})$ is continuously differentiable.
- $f(\mathbf{x})$ is convex.
- $\exists {\bf x^ *} \in {\rm dom}f$, s.t. $ f({\bf x^ *})= f^ * = \min f({\bf x}) $.
- The gradient of $f$ is Lipschitz continuous, that is, $ \Vert \nabla f({\bf x}) - \nabla f({\bf y}) \Vert \le L \Vert {\bf x}-{\bf y} \Vert$, $L>0$.

##### Algorithm:  Gradient Descent
$$
\begin{align}
& \text{Input: An initial point } {\bf x_0} \text{, a constant } \alpha \in (0, \dfrac{2}{L}), \ k = 0 \\\\
& \text{while the termination condition does not hold,  do} \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ k = k + 1 \\\\
& ~ ~ ~ ~ ~ ~ ~ ~ {\bf x_{k+1}}={\bf x_k}-\alpha\nabla f({\bf x_k}) \\\\
& \text{end while}
\end{align}
$$

##### Convergence Rate

**Definition:** Suppose that the sequence { $a_k$} converges to a number $L$. Then, the sequence is said to converge linearly to $L$ if there exists a number $\mu \in (0, 1)$, s.t. $\lim \limits _ {k \to \infty } \dfrac{\vert a_{k + 1} - L\vert}{\vert a_k - L \vert} = \mu$.

**Lemma 1:** Suppose that a function $f \in  C^1$. If $\nabla f$ is Lipschitz continuous with Lipschitz constant $L$, then 
$$
f({\bf y}) \le f({\bf x}) + \langle \nabla f({\bf x}),{\bf y} - {\bf x} \rangle+\dfrac{L}{2} \Vert {\bf y}-{\bf x}\Vert^2.
$$

**Proof:** 
$$
\begin{align}
f({\bf y})-f({\bf x}) &= \int^{\bf y}_{\bf x}\nabla f({\bf z}){\bf dz}
\\\\&= \int^1_0 \langle\nabla f({\bf x}+t({\bf y}-{\bf x})),{\bf y}-{\bf x}\rangle dt
\\\\&= \langle\nabla f({\bf x}),{\bf y}-{\bf x}\rangle+\int^1_0 \langle\nabla f({\bf x}+t({\bf y}-{\bf x}))-\nabla f({\bf x}),{\bf y}-{\bf x}\rangle dt
\\\\&\le \langle\nabla f({\bf x}),{\bf y}-{\bf x}\rangle+\int^1_0 \Vert\nabla f({\bf x}+t({\bf y}-{\bf x}))-\nabla f({\bf x})\Vert\Vert{\bf y}-{\bf x}\Vert dt
\\\\&\le \langle\nabla f({\bf x}),{\bf y}-{\bf x}\rangle+L\Vert{\bf y}-{\bf x}\Vert^2\int^1_0 t dt
\\\\&=\langle\nabla f({\bf x}),{\bf y}-{\bf x}\rangle+\frac{L}{2}\Vert {\bf y}-{\bf x}\Vert^2
\end{align}
$$

（与凹凸性无关）

**Lemma 2 （Descent Lemma） :** Suppose that a function $f \in  C^1$. If $\nabla f$ is Lipschitz continuous with Lipschitz constant $L>0$, then $\forall \\{ {\bf x_k} \\}$ generated by the Gradient Descent Algorithm satisfies
$$
f({\bf x_{k+1}})\le f({\bf x_k})-{\alpha}(1-\frac{L\alpha}{2})\Vert\nabla f({\bf x_k})\Vert^2.
$$

（这也是为什么算法约定$\alpha \in (0, \dfrac{2}{L})$）

下面证明算法可以收敛到最小值，在前提条件下，可以考虑证明：
$$
\lim_{k \to \infty}\nabla f({\bf x_k}) = \nabla f(\lim_{ k\to \infty}{\bf x_k})=0.
$$
**Proof:**

由Lemma 2，
$$
\begin{align}
\Vert \nabla f({\bf x_k})\Vert^2 & \le \frac{f({\bf x_{k}})-f({\bf x_{k+1}})}{\alpha(1-\frac{L\alpha}{2})} \\\\
\sum_k\Vert \nabla  f({\bf x_k})\Vert^2 &\le \frac{f({\bf x_0})-f({\bf x_{k+1}})}{\alpha(1-\frac{L\alpha}{2})}\\\\
&\le\frac{f({\bf x_0})-f^ *}{\alpha(1-\frac{L\alpha}{2})}
\end{align}
$$
这个求和存在固有上界，故
$$
\lim_{k\to \infty} \nabla f({\bf x_k}) =0
$$

##### Efficiency and limitations

**Theorem:**  Consider the Problem（$\min f(x)$）and the sequence generated by the Gradient Descent Algorithm. Then the sequence value $f({\bf x_k})$ tends to the optimum function value in a rate of $O(\frac{1}{k})$.

1. If $\alpha \in (0, \dfrac{1}{L})$
$$
f({\bf x_k})-f^ * \le \frac{1}{k}(\frac{1}{2\alpha}\Vert {\bf x_0-x^ *}\Vert^2)
$$

2. If $ \alpha \in (\dfrac{1}{L}, \dfrac{2}{L}) $
$$
f({\bf x_k})-f^ * \le \frac{1}{k}(\frac{1}{2\alpha}\Vert {\bf x_0-x^ *}\Vert^2+\frac{L\alpha -1}{2-L\alpha}(f({\bf x_0})-f({\bf x ^  * })) ) 
$$

**Proof:**

As $ {\bf x_{k+1}}={\bf x_k}-\alpha\nabla f({\bf x_k})$ and $ f({\bf y}) \le f({\bf x}) + \langle\nabla f({\bf x}),{\bf y}-{\bf x}\rangle+\dfrac{L}{2}\Vert {\bf y}-{\bf x}\Vert^2 $, 

$$
f({\bf x_{k+1}})\le f({\bf x_{k}})-(\frac{1}{\alpha}-\frac{L}{2})\Vert{\bf x_{k+1}-x_k}\Vert^2
$$

$$
f({\bf x_{k+1}})-f^ *\le f({\bf x_{k}})-f^ *-(\frac{1}{\alpha}-\frac{L}{2})\Vert{\bf x_{k+1}-x_k}\Vert^2
$$

Consider the convexity of $f$,

$$
\begin{align}
f({\bf x _ {k + 1} })-f({\bf x^ *}) & \le \langle \nabla f({\bf x_k}), { \bf x_k}-{\bf x^ *}\rangle-(\frac{1}{\alpha}-\frac{L}{2})\Vert{\bf x_{k+1}-x_k}\Vert^2
\\\\ & = -\frac{1}{\alpha} \langle{\bf x_{k+1}}-{\bf x_k},{\bf x_k}-{\bf x^ *}\rangle-(\frac{1}{\alpha}-\frac{L}{2})\Vert{\bf x_{k+1}-x_k}\Vert^2
\\\\ & = -\frac{1}{2\alpha}(\Vert{\bf x_{k+1}}-{\bf x^ *}\Vert^2-\Vert{\bf x_{k+1}}-{\bf x_k}\Vert^2-\Vert{\bf x_{k}}-{\bf x^ *}\Vert^2)-(\frac{1}{\alpha}-\frac{L}{2})\Vert{\bf x_{k+1}}-{\bf x_k}\Vert^2
\\\\ & = \frac{1}{2\alpha}(\Vert{\bf x_{k}}-{\bf x^ *}\Vert^2-\Vert{\bf x_{k+1}}-{\bf x^ *}\Vert^2)-(\frac{1}{2\alpha}-\frac{L}{2})\Vert{\bf x_{k+1}}-{\bf x_k}\Vert^2
\end{align}
$$

Summing up the inequalities,

$$
\begin{align}
k(f({\bf x_{k} })-f({\bf x ^ *})) &\le \sum ^ { k-1 } _ { i=0 } ( f({\bf x _ {i + 1}})- f({\bf x^ * }) ) \\\\
& \le \frac{1}{2 \alpha }(\Vert {\bf x_{0} }-{\bf x^ *}\Vert^2-\Vert{\bf x_{k}} - {\bf x ^ *} \Vert^2 ) - (\frac{1}{2\alpha} - \frac{L}{2} ) \sum^{k-1}_{i=0} \Vert{\bf x _ {i+1} }- {\bf x_i} \Vert^2
\end{align}
$$

1. If $\alpha \in (0, \dfrac{1}{L})$, $\dfrac{1}{2\alpha}-\dfrac{L}{2}>0$, then

$$
k(f({\bf x_{k}})-f({\bf x^ *})) \le\frac{1}{2\alpha}\Vert{\bf x_{0}}-{\bf x^ *}\Vert^2.
$$

2. If $\alpha \in (\dfrac{1}{L}, \dfrac{2}{L})$, $\dfrac{1}{2\alpha}-\dfrac{L}{2}>0$, then

$$
\begin{align}
k (f({\bf x_k })-f({\bf x^ *})) & \le \frac{1}{2 \alpha} (\Vert { \bf x_0} - {\bf x^ *} \Vert ^2 - \Vert{\bf x_k}- {\bf x^ *} \Vert^2) + \frac {L \alpha-1}{2\alpha}\sum^{k-1} _ { i = 0 } \Vert {\bf x _ {i+1} } - {\bf x_i} \Vert^2
\\\\ & \le \frac{1}{2 \alpha} \Vert { \bf x_0} - { \bf x^ * } \Vert^2 + \frac{L \alpha-1}{2 \alpha} \sum^{ \infty} _ {i=0} \Vert {\bf x _ {i+1} } - {\bf x_i} \Vert^2
\\\\ & \le  \frac{1}{2 \alpha} \Vert {\bf x_0} - {\bf x^ *} \Vert^2 + \frac{L \alpha-1}{2 \alpha}\ \frac{2\alpha}{2-L\alpha} (f({\bf x_0})-f({\bf x^ *})) (Lemma \ 2)
\end{align}
$$

Remark: $\Vert {\bf x_k} - {\bf x^ *} \Vert$ doesn't always converge to 0.
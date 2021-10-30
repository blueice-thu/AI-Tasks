# 人工智能导论第三次作业

## 1

$$
\sum_{i=1}^{n} \log p(\boldsymbol{x} \mid \mu, \boldsymbol{\Sigma})=-\frac{n}{2} \log |2 \pi \boldsymbol{\Sigma}|-\frac{1}{2} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\mu\right)
$$

(1) 令 $\nabla_{\mu}J(\mu,\boldsymbol{\Sigma})=0$ ，即：
$$
\begin{align*}
\frac{\partial\sum_{i=1}^{n} \log p(\boldsymbol{x} \mid \mu, \boldsymbol{\Sigma})}{\partial\mu}&=0\\
\frac{\partial\left[-\frac{n}{2} \log |2 \pi \boldsymbol{\Sigma}|-\frac{1}{2} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\mu\right)\right]}{\partial\mu}&=0\\
\frac{\partial\left[\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\mu\right)\right]}{\partial\mu}&=0\\
\boldsymbol{\Sigma}^{-1}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)&=0\\
\mu&=\frac{1}{n}\sum_{i=1}^{n}\boldsymbol{x}_{i}
\end{align*}
$$
故 $\hat{\mu}_{\text{MLE}}=\frac{1}{n}\sum_{i=1}^{n}\boldsymbol{x}_{i}$ 。另一方面，令 $\nabla_{\Sigma}J(\mu,\boldsymbol{\Sigma})=0$ ，即：
$$
\begin{align*}
\frac{\partial\sum_{i=1}^{n} \log p(\boldsymbol{x} \mid \mu, \boldsymbol{\Sigma})}{\partial\boldsymbol{\Sigma}}&=0\\

\frac{\partial\left[-\frac{n}{2} \log |2 \pi \boldsymbol{\Sigma}|-\frac{1}{2} \sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}_{i}-\mu\right)\right]}{\partial\boldsymbol{\Sigma}}&=0\\

-\frac{n}{2}\boldsymbol{\Sigma}^{-1}+\frac{1}{2}\boldsymbol{\Sigma}^{-1}\left(\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)^{T}\left(\boldsymbol{x}_{i}-\mu\right)\right) \boldsymbol{\Sigma}^{-1}&=0\\

n\boldsymbol{\Sigma}&=\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)\left(\boldsymbol{x}_{i}-\mu\right)^{T}\\

\boldsymbol{\Sigma}&=\frac{1}{n}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\mu\right)\left(\boldsymbol{x}_{i}-\mu\right)^{T}

\end{align*}
$$
故 $\hat{\Sigma}_{\text{MLE}}=\frac{1}{n}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\hat{\mu}_{\text{MLE}}\right)\left(\boldsymbol{x}_{i}-\hat{\mu}_{\text{MLE}}\right)^{T}$ 。

(2) 证明：
$$
\begin{align*}
\text{E}\left[\hat{\boldsymbol{\Sigma}_{\text{MLE}}}\right]&=\text{E}\left[\frac{1}{n}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}-\hat{\mu}_{\text{MLE}}\right)\left(\boldsymbol{x}_{i}-\hat{\mu}_{\text{MLE}}\right)^{T}\right]\\
&=\text{E}\left[\frac{1}{n}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}\boldsymbol{x}_{i}^T-\hat{\mu}_{\text{MLE}}\boldsymbol{x}_{i}^T-\boldsymbol{x}_{i}\hat{\mu}_{\text{MLE}}^T+\hat{\mu}_{\text{MLE}}\hat{\mu}^T_{\text{MLE}}\right)\right]\\
&=\text{E}\left[\frac{1}{n}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}\boldsymbol{x}_{i}^T\right)-\hat{\mu}_{\text{MLE}}\hat{\mu}^T_{\text{MLE}}\right]\\
&=\text{E}\left[\frac{1}{n}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}\boldsymbol{x}_{i}^T\right)-\mu\mu^T+\mu\mu^T-\hat{\mu}_{\text{MLE}}\hat{\mu}^T_{\text{MLE}}\right]\\
&=\text{E}\left[\frac{1}{n}\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}\boldsymbol{x}_{i}^T\right)-\mu\mu^T\right]-\text{E}\left(\hat{\mu}_{\text{MLE}}\hat{\mu}^T_{\text{MLE}}-\mu\mu^T\right)\\
&=\text{E}\left[\frac{\sum_{i=1}^{n}\left(\boldsymbol{x}_{i}\boldsymbol{x}_{i}^T-\mu\mu^T\right)}{n}\right]-\left[\text{E}\left(\hat{\mu}_{\text{MLE}}\hat{\mu}^T_{\text{MLE}}\right)-\mu\mu^T\right]\\
&=\boldsymbol{\Sigma}-\left[\text{E}\left(\hat{\mu}_{\text{MLE}}\hat{\mu}^T_{\text{MLE}}\right)-\text{E}\left(\hat{\mu}_{\text{MLE}}\right)\text{E}^T\left(\hat{\mu}_{\text{MLE}}\right)\right]\\
&=\boldsymbol{\Sigma}-\text{Var}\left(\hat{\mu}_{\text{MLE}}\right)\\
&=\boldsymbol{\Sigma}-\text{Var}\left(\frac{1}{n}\sum_{i=1}^{n}\boldsymbol{x_i}\right)\\
&=\boldsymbol{\Sigma}-\frac{1}{n^2}\sum_{i=1}^{n}\text{Var}\left(\boldsymbol{x_i}\right)\\
&=\boldsymbol{\Sigma}-\frac{1}{n}\boldsymbol{\Sigma}\\
&=\frac{n-1}{n}\boldsymbol{\Sigma}
\end{align*}
$$

## 2

(1) 

<img src="D:\MyProject\Introduction-to-Artificial-Intelligence\hw3\imgs\hw3_2.png" alt="hw3_2" style="zoom: 80%;" />

$$
\begin{align*}
&p(S_1,S_2,S_3,S_4,R_1,R_2,A_1,A_2)\\
=&p(S_1|R_2)p(S_2|R_2)p(S_3|R_1,R_2)p(S_4|R_1,R_2)p(R_1|A_2)p(R_2|A_1)p(A_1)p(A_2)
\end{align*}
$$
(2) $R_1, R_2$

(3) 需要 $2+2+4+4+2+2+1+1=18$ 个参数。

取消独立性假设：$2^8-1=255$ 个参数。

(4) 若有吸烟经历，则 $R_1$ 的概率发生变化；若已知 $S_3=1$ ，则 $R_1,R_2$ 的概率都发生变化。

(5) 
$$
\begin{align*}
&p(R_2|A_1,A_2,S_1,S_2,S_3,S_4)\\
=&\frac{p(A_1,A_2,S_1,S_2,S_3,S_4,R_2)}{p(A_1,A_2,S_1,S_2,S_3,S_4)}\\
=&\frac{\sum_{R_1}p(A_1,A_2,S_1,S_2,S_3,S_4,R_1,R_2)}{\sum_{R_1,R_2}p(A_1,A_2,S_1,S_2,S_3,S_4,R_1,R_2)}\\
=&\frac{\sum_{R_1}p(S_1|R_2)p(S_2,R_2)p(S_3|R_1,R_2)p(S_4|R_1,R_2)p(R_1|A_2)p(R_2|A_1)p(A_1)p(A_2)}{\sum_{R_1,R_2}p(S_1|R_2)p(S_2,R_2)p(S_3|R_1,R_2)p(S_4|R_1,R_2)p(R_1|A_2)p(R_2|A_1)p(A_1)p(A_2)}\\
=&\frac{\sum_{R_1}p(S_1|R_2)p(S_2,R_2)p(S_3|R_1,R_2)p(S_4|R_1,R_2)p(R_1|A_2)p(R_2|A_1)}{\sum_{R_1,R_2}p(S_1|R_2)p(S_2,R_2)p(S_3|R_1,R_2)p(S_4|R_1,R_2)p(R_1|A_2)p(R_2|A_1)}\\
\end{align*}
$$

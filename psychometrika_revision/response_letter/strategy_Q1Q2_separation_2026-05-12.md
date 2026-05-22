# Q1/Q2分離 実装作戦書

**作成日:** 2026-05-12
**対象:** PSY-2025-0213 大改訂版 R2-2a への応答
**範囲:** `overleaf_manuscript/` における理論記述のアップデート方針

---

## 0. 目的と射程

R2-2a で指摘された ``observability problem'' に対し、レビュアーが推奨する
multilevel decomposition (Muthén 1994; Asparouhov et al. 2018) を
manuscript の実装理論 (Section 2.3, Section 3) に正式に取り込む。

レビュアーの本質的主張:

> ``Elements such as η_{2i} and ζ_{2i} are time invariant so it is generally
> unnecessary to put any of them into the extended Kim filter portion of the
> algorithm. In fact, doing so and arbitrarily adding diffuse or slightly
> misspecified initial conditions for the latent variables would actually
> negatively impact the quality of the estimation results.''

我々の現状:

| 時間不変成分 | 現在の扱い | 観測可能性 | 状態 |
|---|---|---|---|
| η_{2i} (between レベルの因子スコア) | Bartlett で Kim filter の外で推定 (Sec. 3.2 / Alg. 1 step 2) | OK | 解決済 |
| ζ_{2i} (個人別ランダム切片の残差) | augmented state ベクトル `η_{aug,it} = [η_{1it}; ζ_{2i}]` に同居 (A3) | NOT OK | 未解決 |

よって作戦の中核は **ζ_{2i} を Kim filter の外に出すための二段階分離手法を
manuscript に明文化すること** である。

---

## 1. 理論的バックボーン

### 1.1 Muthén (1994) の二水準共分散構造分解

Muthén (1994, pp. 379–386) の中心式:

$$
\Sigma_T \;=\; \Sigma_W + \Sigma_B
$$

ここで

- $\Sigma_W = \Lambda_W \Psi_W \Lambda_W^\top + \Theta_W$
  (within 共分散; pooled within sample covariance $S_{PW}$ で不偏推定)
- $\Sigma_B = \Lambda_B \Psi_B \Lambda_B^\top + \Theta_B$
  (between 共分散; $S_B - S_{PW}$ をグループサイズで割って一致推定)

すなわち、

$$
S_{PW} \;\to\; \Sigma_W, \qquad
c^{-1}(S_B - S_{PW}) \;\to\; \Sigma_B \quad (c = \text{group size}).
$$

FIML は両者を同時に尤度に組み込む。MUML は二群 SEM の形に書き直して既存
ソフトで解く近似 ML 推定量 (式 (20))。本論文の文脈ではどちらでも実装可能だが
**FIML 系の同時推定** を理論上の参照点とし、現実的実装としては **二段階
(REML 類似) 手法** を採用するという二段構えで記述するのが最も筋が通る。

### 1.2 DSEM の latent centering (Asparouhov, Hamaker, Muthén 2018)

DSEM では各被験者ごとに観測値を

$$
y_{1it} \;=\; \mu_i + y^{(w)}_{1it}, \qquad \mu_i \perp y^{(w)}_{1it}
$$

と分解し、$\mu_i$ を between レベルの潜在変数として、$y^{(w)}_{1it}$ を
within レベルの動的モデルへ渡す。$\mu_i$ は ``time invariant'' なので
Kalman filter のような within フィルタ内には入れないというのが latent
centering の基本思想であり、これは Muthén (1994) の within/between 分解を
時系列文脈で発展させたものに対応する。

### 1.3 REML (Snijders & Bosker 2012, Ch. 4)

線形混合効果モデルにおける REML は、固定効果を消去した残差から分散成分を
推定する。同時 ML が固定効果の不確実性を吸収できないために分散成分を
下向きにバイアスするのに対し、REML は射影によって自由度を補正する。

本論文への適応:

- 固定効果 (within レベルの構造母数 $\bm{B}_{3is}, \bm{Q}_{1s}, \bm{R}_{1s}$,
  regime swithcing 母数 $\bm{\gamma}$) は Kim filter で最大化
- 分散成分 $\bm{Q}_2$ と個人別残差 $\bm{\zeta}_{2i}$ は個人平均 (固定効果の
  ``射影残差'') から推定

これは REML の精神的アナロジーであって、厳密な REML ではない (後述)。

---

## 2. 提案するアルゴリズム改訂

### 2.1 ζ_{2i} を augmented state から外す

augmented state を簡素化:

$$
\bm{\eta}_{\text{aug},it} \;\equiv\; \bm{\eta}_{1it}
\quad (U_{\text{aug}} = U_1)
$$

これにより:

- 状態遷移行列 $\bm{B}_{3is, \text{aug}}$ は $U_1 \times U_1$ の $\bm{B}_{3is}$ そのもの
- $\bm{Q}_{1s,\text{aug}} = \bm{Q}_{1s}$
- $\bm{\Lambda}_{1s, \text{aug}} = \bm{\Lambda}_{1s}$
- $\bm{\zeta}_{2i}$ の diffuse 初期値の問題は消失
- observability 問題が消失 (B_{3is} が stationary の標準仮定で OK)

ただし、within の状態方程式は依然として ``intercept'' に $\bm{\zeta}_{2i}$ を
含む。それを Kim filter に外部から ``plug in'' する形にする。

### 2.2 二段階推定 (一回パス)

**Stage A. Between レベル (個人平均を使う)**

欠測を許すため、person $i$ について観測された時点集合を $\text{obs}(i)$、
その要素数を $T_i^{\text{obs}} = |\text{obs}(i)|$ と書く。各被験者の
time average を計算:

$$
\bar{\bm{y}}_{1i\cdot} \;=\; \frac{1}{T_i^{\text{obs}}} \sum_{t \in \text{obs}(i)} \bm{y}_{1it}
$$

次に、within-level state equation

$$
(\bm{\eta}_{1it}\mid S_{it}=s)
= \bm{b}_{1is} + \bm{B}_{3is}\bm{\eta}_{1i,t-1} + \bm{\zeta}_{1its},
\qquad
\bm{b}_{1is} = \bm{b}_{1s} + \bm{B}_{2s}\bm{\eta}_{2i} + \bm{\zeta}_{2i}
$$

から、regime $s$ の下での **within-process の定常平均** を導く。これは
``within を引いた残り'' ではなく、VAR(1) 過程の長期平均
$\bm{\mu}_{is} = \mathbb{E}(\bm{\eta}_{1it}\mid S_{it}=s,\bm{\eta}_{2i},\bm{\zeta}_{2i})$
を求めているだけである。stationary VAR(1) 仮定のもとで
$\mathbb{E}(\bm{\zeta}_{1its})=\bm{0}$ および
$\mathbb{E}(\bm{\eta}_{1it}\mid\cdots)=\mathbb{E}(\bm{\eta}_{1i,t-1}\mid\cdots)=\bm{\mu}_{is}$
とおけば、

$$
(\bm{I}-\bm{B}_{3is})\bm{\mu}_{is}
= \bm{b}_{1s} + \bm{B}_{2s}\bm{\eta}_{2i} + \bm{\zeta}_{2i},
$$

よって regime $s$ に属する期間の長期平均は

$$
\mathbb{E}[\bm{\eta}_{1it} \mid S_{it}=s, \bm{\eta}_{2i}, \bm{\zeta}_{2i}]
= (\bm{I} - \bm{B}_{3is})^{-1} \bigl(\bm{b}_{1s} + \bm{B}_{2s}\bm{\eta}_{2i} + \bm{\zeta}_{2i}\bigr).
$$

これはスカラー AR(1) の定常平均
$\mathbb{E}(x_t)=a/(1-\phi)$ の多変量版に相当する。

ここで注意すべき点が二つある。第一に、上の定常平均の式は理論上は
``もし真の between-level latent variable $\bm{\eta}_{2i}$ が分かっていれば''
という形で書いている。しかし実際のアルゴリズムでは $\bm{\eta}_{2i}$ は観測
できないため、step 2 の CFA / Bartlett step で得た
$\hat{\bm{\eta}}_{2i}$ をその代理として代入する。すなわち、理論式では
$\bm{\eta}_{2i}$ を条件変数として書き、実装ではその推定値
$\hat{\bm{\eta}}_{2i}$ を plug-in する。

第二に、Stage A で実際にデータとして持っているのは latent state
$\bm{\eta}_{1it}$ ではなく、その observed proxy $\bm{y}_{1it}$ の個人平均
$\bar{\bm{y}}_{1i\cdot}$ である。したがって、最終的に必要なのは latent mean
そのものではなく、measurement model を通して latent mean を観測空間へ写した
``person $i$ の観測平均 $\bar{\bm{y}}_{1i\cdot}$ の model-implied mean'' である。
そのため

$$
\bm{m}_i(\bm{\theta}, \bm{\zeta}_{2i})
\;\approx\;
\mathbb{E}\!\left(
\bar{\bm{y}}_{1i\cdot}
\mid
\hat{\bm{\eta}}_{2i},
\bm{\zeta}_{2i},
\mathcal{D}_{1:T_i^{\text{obs}}}
\right).
$$

以下では $\bm{m}_i(\bm{\theta}, \bm{\zeta}_{2i})$ を **person $i$ の観測平均
$\bar{\bm{y}}_{1i\cdot}$ の model-implied mean** として固定し、manuscript の
既存 notation で展開すると、

$$
\bm{m}_i(\bm{\theta}, \bm{\zeta}_{2i})
= \frac{1}{T_i^{\text{obs}}}
\sum_{t\in\text{obs}(i)}
\sum_{s=1}^{2}
w_{its}
\bm{\Lambda}_{1s}
(\bm{I}-\bm{B}_{3is})^{-1}
\bigl(\bm{b}_{1s} + \bm{B}_{2s}\hat{\bm{\eta}}_{2i} + \bm{\zeta}_{2i}\bigr).
$$

すなわち、各時点 $t$ における regime-specific な観測平均
$\mathbb{E}(\bm{y}_{1it}\mid S_{it}=s,\hat{\bm{\eta}}_{2i},\bm{\zeta}_{2i})$ を
regime weight $w_{its}$ で重み付けし、それを個人内で平均したものが
$\bm{m}_i(\bm{\theta}, \bm{\zeta}_{2i})$ である。Stage A の最小化問題:

$$
\min_{\{\bm{\zeta}_{2i}\}, \bm{Q}_2}
\sum_{i=1}^{N}
\bigl[\bar{\bm{y}}_{1i\cdot} - \bm{m}_i(\bm{\theta}, \bm{\zeta}_{2i})\bigr]^\top
\bm{V}_{\bar{y},i}^{-1}
\bigl[\bar{\bm{y}}_{1i\cdot} - \bm{m}_i(\bm{\theta}, \bm{\zeta}_{2i})\bigr]
+ \log|\bm{V}_{\bar{y},i}|
+ \bm{\zeta}_{2i}^\top \bm{Q}_2^{-1} \bm{\zeta}_{2i} + \log|\bm{Q}_2|
$$

ここでは ``個人平均の周辺尤度'' と言い切るより、**between-person Gaussian
objective** と呼ぶ方が安全である。厳密な fully integrated likelihood ではなく、
current iteration の regime weight $w_{its}$ と $\hat{\bm{\eta}}_{2i}$ を
plug-in したガウス型の penalized objective とみなすのが実態に近い。これは
Muthén (1994) で言う $S_B$ の解析に対応する。点推定 $\hat{\bm{\zeta}}_{2i}$ は
posterior mode で、empirical Bayes 予測子として与えられる。

**Stage B. Within レベル (Kim filter, 個人中心化観測)**

中心化:

$$
\tilde{\bm{y}}_{1it} \;=\; \bm{y}_{1it} - \bar{\bm{y}}_{1i\cdot}
$$

これに対する状態方程式は (近似的に)

$$
\bm{\eta}_{1it} - \mathbb{E}[\bm{\eta}_{1it}\mid i] \;=\; \bm{B}_{3is} \bigl(\bm{\eta}_{1i,t-1} - \mathbb{E}[\bm{\eta}_{1i,t-1}\mid i]\bigr) + \bm{\zeta}_{1its},
$$

すなわち $\bm{\zeta}_{2i}$ は中心化により消える。Kim filter は単純な
$\bm{\eta}_{1it}$ の filter として動かす ($\bm{\eta}_{\text{aug},it} = \bm{\eta}_{1it}$)。これは
Muthén (1994) で言う $S_{PW}$ の解析に対応する。

**Stage C. 統合**

Stage A と B の結果を統合して全パラメタ $\bm{\theta} = \{\bm{b}_{1s},\bm{B}_{2s}, \bm{B}_{3s},\bm{B}_{4s}, \bm{Q}_{1s}, \bm{Q}_2, \bm{R}_{1s}, \bm{\gamma}_{s'\cdot}, \dots\}$ を更新。
**B → A → B → ...** の交互反復 (profile likelihood 様) で収束させる。
収束時の SE は通常の OPG / 数値 Hessian で計算可能。
これが本論文での提案実装である。

### 2.3 完全 FIML / Kalman EM (将来研究)

Shumway & Stoffer (1982) の EM 枠組では、$\bm{\zeta}_{2i}$ を完全データの
一部とみなし、E-step で smoothed 期待値を計算、M-step で $\bm{Q}_2$ を更新
する。これにより Stage A と B が一つの likelihood の中で正式に統合される。
本論文の射程外とし、Discussion / Open questions に明示。

---

## 3. Manuscript 改訂方針 (具体的箇所)

### 3.1 `2. models.tex` (Section 2.3 Between-level structural model)

**現状:** $\bm{\zeta}_{2i} \sim \mathcal{N}(\bm{0}, \bm{Q}_2)$ を定義する記述あり。

**追記:** Equation (BSM) の直後に、次のような注釈を追加 (赤字 = 新規):

> Note that although Eq.~(\ref{BSM}) introduces $\bm{\zeta}_{2i}$ as a
> time invariant random intercept residual, in our estimation procedure
> (Section~\ref{sec:estimation}) $\bm{\zeta}_{2i}$ is not treated as part
> of the dynamic state vector. Rather, it is estimated together with
> $\bm{Q}_2$ from person averages of the within level observations, in a
> manner analogous to the between within decomposition of
> \citet{Muthen1994} and the latent centering procedure of
> \citet{Asparouhov2018}.

### 3.2 `3. estimation.tex` (Algorithm 1 全体構造)

**現状:** Algorithm 1 は

1. Initialization
2. CFA (Bartlett η_{2i})
3. Extended Kalman filter (augmented state contains ζ_{2i})
4. Extended Hamilton filter
5. Collapsing
6. Parameter update

**改訂:** ステップ 2 と 3 の間に **新ステップ 2b ``Between level random intercept extraction''** を挿入し、Algorithm 1 を以下に差し替える:

```
1: Initialization
While not converged:
  2 : CFA (Bartlett scores  η̂_{2i})
  2b: Between level update (compute ȳ_{1i.}, estimate  ζ̂_{2i} and  Q̂_2)
  For t = 1, ..., T:
    3 : Extended Kalman filter on   ỹ_{1it} = y_{1it} - ȳ_{1i.}
    4 : Extended Hamilton filter
    5 : Collapsing process
  6 : Parameter update
```

そして augmented state を非拡張に変更:
$\bm{\eta}_{\text{aug},it} := \bm{\eta}_{1it}$ ($U_\text{aug}=U_1$),
$\bm{B}_{3is,\text{aug}} := \bm{B}_{3is}$, etc.

旧 augmented 構造 (block matrix with $\bm{I}$) は **将来研究としての``unified Kalman EM''実装** の文脈で Discussion か Appendix に移動。

### 3.3 `3. estimation.tex` に新サブセクションを追加

**重要: numbering の整合性確認.** Response letter (R2-2a 応答) は
``in the revised Section~3.3'' と書いている。現行の節構成は

- 3.1 Initialization
- 3.2 Confirmatory factor analysis
- 3.3 Extended Kalman filter
- 3.4 Extended Hamilton filter
- 3.5 Collapsing
- 3.6 Parameter update
- 3.7 Missing values

なので、新サブセクションを **新 3.3 ``Between level random intercept extraction''** として挿入し、旧 3.3 以降を 3.4 以降にシフトする。Response letter
の言及位置と一致する。

`\subsection{The confirmatory factor analysis (step 2)}` の後ろに、次の新サブセクションを追加:

> **\subsection{Between level random intercept extraction (step 2b)}**
>
> The person specific random intercept residual $\bm{\zeta}_{2i}$ is
> time invariant; carrying it inside the dynamic state vector would
> render the augmented state space non observable (cf.\ R2 2a;
> \citealp{Asparouhov2018}). Following the multilevel decomposition of
> \citet{Muthen1994}, we therefore estimate $\bm{\zeta}_{2i}$ outside
> the Kim filter, in parallel with the Bartlett step.
>
> Let
> $\bar{\bm{y}}_{1i\cdot} = (T_i^{\text{obs}})^{-1} \sum_{t \in \text{obs}(i)} \bm{y}_{1it}$
> denote person~$i$'s sample mean of the within level observations,
> where $\text{obs}(i)$ is the set of observed time points and
> $T_i^{\text{obs}} = |\text{obs}(i)|$.
> Under stationarity of $\bm{B}_{3is}$, the regime specific long run
> mean of the within level latent variables is obtained from
> $\bm{\eta}_{1it} = (\bm{b}_{1s} + \bm{B}_{2s}\bm{\eta}_{2i} +
> \bm{\zeta}_{2i}) + \bm{B}_{3is}\bm{\eta}_{1i,t-1} + \bm{\zeta}_{1its}$
> by taking conditional expectations and solving the fixed point
> equation
>
> \begin{equation}
>   \mathbb{E}(\bm{\eta}_{1it}\mid S_{it}=s,\bm{\eta}_{2i},\bm{\zeta}_{2i})
>   =
>   (\bm{I} - \bm{B}_{3is})^{-1}
>   (\bm{b}_{1s} + \bm{B}_{2s}\bm{\eta}_{2i} + \bm{\zeta}_{2i}).
> \end{equation}
>
> This is the stationary mean of the regime specific VAR(1) process,
> not a residual quantity after removing the within person variation.
> Two distinctions are important here. First, the equation above is
> written at the population level as if the true between-level latent
> variable $\bm{\eta}_{2i}$ were known. In the actual algorithm,
> however, $\bm{\eta}_{2i}$ is unobserved and is replaced by the
> Bartlett estimate $\hat{\bm{\eta}}_{2i}$ obtained in
> step~\textcolor{red}{2}. Second, step~2b does not operate on the
> latent states $\bm{\eta}_{1it}$ directly. The available data are the
> observed variables $\bm{y}_{1it}$, summarized by the person mean
> $\bar{\bm{y}}_{1i\cdot}$. Accordingly, the object required in
> step~2b is not the latent mean itself, but the model-implied mean of
> the observed person average $\bar{\bm{y}}_{1i\cdot}$, that is,
>
> \begin{equation}
>   \bm{m}_i(\bm{\theta},\bm{\zeta}_{2i})
>   \approx
>   \mathbb{E}\!\left(
>   \bar{\bm{y}}_{1i\cdot}
>   \mid
>   \hat{\bm{\eta}}_{2i},
>   \bm{\zeta}_{2i},
>   \mathcal{D}_{1:T_i^{\text{obs}}}
>   \right).
> \end{equation}
>
> Using regime weights $w_{its}$ from the current iteration, this model
> implied mean is approximated by
>
> \begin{equation} \label{ImpliedMean}
>   \bm{m}_i(\bm{\theta},\bm{\zeta}_{2i})
>   = \frac{1}{T_i^{\text{obs}}}
>     \sum_{t \in \text{obs}(i)}
>     \sum_{s=1}^{2}
>     w_{its}
>     \bm{\Lambda}_{1s}
>     (\bm{I} - \bm{B}_{3is})^{-1}
>     \bigl(\bm{b}_{1s} + \bm{B}_{2s} \hat{\bm{\eta}}_{2i} + \bm{\zeta}_{2i}\bigr),
> \end{equation}
>
> where $w_{its}$ denotes a regime weight for person~$i$, time~$t$, and
> regime~$s$. In the current filter based implementation, the natural
> choice is the updated Hamilton filter probability
> $w_{its} = Pr(S_{it}=s \mid \mathcal{D}_{1:t})$ from Equation~(\ref{HF4}).
> If a smoothing step is added in a future implementation, these weights
> can be replaced by full sample probabilities
> $w_{its} = Pr(S_{it}=s \mid \mathcal{D}_{1:T_i^{\text{obs}}})$.
>
> The empirical Bayes estimator of $\bm{\zeta}_{2i}$ and the moment
> estimator of $\bm{Q}_2$ are obtained from the between person normal
> equations
>
> \begin{align} \label{ZetaEB}
>   \hat{\bm{\zeta}}_{2i} &= \bm{Q}_2
>     \bm{D}_i^\top \bm{V}_{\bar{y},i}^{-1}
>     \bigl(\bar{\bm{y}}_{1i\cdot} - \bm{m}_i(\bm{\theta}, \bm{0})\bigr), \\
>   \hat{\bm{Q}}_2 &= \frac{1}{N} \sum_{i=1}^{N}
>     \hat{\bm{\zeta}}_{2i} \hat{\bm{\zeta}}_{2i}^\top + \bm{C}_i,
> \end{align}
>
> where $\bm{D}_i = \partial\bm{m}_i / \partial \bm{\zeta}_{2i}^\top$,
> $\bm{V}_{\bar{y},i}$ is the model implied covariance of
> $\bar{\bm{y}}_{1i\cdot}$, and $\bm{C}_i$ is the empirical Bayes
> uncertainty correction. Strictly speaking, these equations define a
> between-person Gaussian penalized objective rather than a fully
> integrated likelihood. The same logic underlies the between
> within decomposition of \citet{Muthen1994} ($S_B$ as estimator of
> $\Sigma_W + c\,\Sigma_B$) and the latent centering procedure of
> \citet{Asparouhov2018}. In the multilevel linear model literature,
> conditioning on the fixed effects when estimating variance
> components is the defining feature of restricted maximum
> likelihood \citep[REML;][Ch.~4]{SnijdersBosker2012}, which avoids
> the downward bias of joint ML.
>
> The Kim filter in steps 3 through 5 is then run on the person
> centered observations
> $\tilde{\bm{y}}_{1it} = \bm{y}_{1it} - \bar{\bm{y}}_{1i\cdot}$, with
> the simplified non augmented state vector $\bm{\eta}_{\text{aug},it}
> \equiv \bm{\eta}_{1it}$ of dimension $U_1$.

### 3.4 `3. estimation.tex` Kalman filter のパラメタ簡素化

augmented state を単純化したことに伴い、Equations
(\ref{Kalman1_aug})~(\ref{Kalman7_aug}) の augmented 行列定義を縮約:

旧
$\bm{b}_{1is,\text{aug}}^{*} = [\bm{b}_{1s} + \bm{B}_{2s}\bm{\eta}_{2i};\; \bm{0}]$
→ 新 $\bm{b}_{1is}^{*} = \bm{b}_{1s} + \bm{B}_{2s} \hat{\bm{\eta}}_{2i}$ (ただし
中心化観測を使うのでこの項は ``個人別 implied mean'' との差分として暗黙に
含まれる; 詳細は新 step 2b 内で処理済み)

旧
$\bm{B}_{3is,\text{aug}} = [[\bm{B}_{3is},\bm{I}];[\bm{0},\bm{I}]]$
→ 新 $\bm{B}_{3is,\text{aug}} = \bm{B}_{3is}$

旧
$\bm{Q}_{1s,\text{aug}} = \mathrm{diag}(\bm{Q}_{1s},\bm{0})$
→ 新 $\bm{Q}_{1s,\text{aug}} = \bm{Q}_{1s}$

旧 $\bm{\Lambda}_{1s,\text{aug}} = [\bm{\Lambda}_{1s},\bm{0}]$
→ 新 $\bm{\Lambda}_{1s,\text{aug}} = \bm{\Lambda}_{1s}$

### 3.5 `A3. estimation.tex` (Augmented state space formulation)

このセクションは大幅縮約。次のように構成し直す:

- **Sub A3.1 ``Original augmented formulation (legacy)''** = 旧 A3 全体を保持
  (歴史的経緯・代替実装としての参考)
- **Sub A3.2 ``Two stage formulation adopted in the present paper''** =
  Section 3 の新 step 2b に対応する詳細導出 (式 (\ref{ImpliedMean}),
  (\ref{ZetaEB}) の導出, 周辺尤度の証明, REML 類似性の導出)
- **Sub A3.3 ``Towards a unified Kalman EM''** = Shumway and Stoffer 1982
  に基づく将来実装の sketch (E step で smoothed $\bm{\zeta}_{2i}$,
  M step で $\bm{Q}_2$ 更新)

### 3.6 `A1. notation.tex` の更新

$\bm{B}_{3is,\text{aug}}, \bm{Q}_{1s,\text{aug}}, \bm{\Lambda}_{1s,\text{aug}}$
の dimension entry を新しい (非拡張) 定義に揃える。
$\bm{A}_i^{(\pi)}, \pi_{is}, \bar{\bm{y}}_{1i\cdot}, \tilde{\bm{y}}_{1it}$
を追加。

### 3.7 `6. discussion.tex` (Open questions)

新 paragraph を追加:

> The two stage procedure introduced in Section~\ref{sec:estimation}
> follows the spirit of REML estimation \citep[Ch.~4]{SnijdersBosker2012}
> by conditioning the variance component estimator $\hat{\bm{Q}}_2$ on
> the within level fixed effects. A fully integrated implementation,
> e.g.\ via the Kalman EM algorithm of \citet{ShumwayStoffer1982}, would
> embed the estimation of $\bm{\zeta}_{2i}$ within a unified likelihood
> objective and replace the iterative profile structure of Algorithm~1
> with a single coherent EM loop. We retain this extension as an
> important direction for future methodological work.

---

## 4. 文献追加 (`*.bib`)

以下を bib に追加 (存在しなければ):

- `Muthen1994` (Multilevel Covariance Structure Analysis, Sociol. Meth. Res.)
- `Asparouhov2018b` (Dynamic SEM technical companion if used)
- `SnijdersBosker2012` (Multilevel Analysis, 2nd ed., Sage)
- `ShumwayStoffer1982` (EM for state space models, J. Time Ser. Anal.)

`Asparouhov2018` (本体) は既に Sec. 2 で引用済 (introduction.tex を参照)。

---

## 5. 実装サニティチェック (本論文 vs Muthén 1994 対応表)

| Muthén (1994) | 本論文 (改訂後) |
|---|---|
| $\Sigma_W$ | within VAR(1) covariance generated by $\bm{B}_{3is}, \bm{Q}_{1s}$ |
| $\Sigma_B$ | $\bm{\Lambda}_2 \bm{P}_2 \bm{\Lambda}_2^\top + \bm{R}_2$ (CFA on $\bm{y}_{2i}$) + $\bm{Q}_2$ (random intercept) |
| $S_{PW}$ | pooled within sample cov of $\tilde{\bm{y}}_{1it}$ |
| $S_B$ | sample cov of $\bar{\bm{y}}_{1i\cdot}$ |
| MUML two group structure | step 2b (between) + steps 3 5 (within) iterated |
| FIML | future work (Kalman EM, Section A3.3) |

---

## 6. 実行手順 (具体的タスク)

実装は今回の改訂サイクル内では **理論記述のみ** に留め、コード変更は次稿で
扱う方針。今回 (改訂版提出, 2026-06-09 締切) で manuscript に入れるのは
以下のテキスト改訂のみ。

1. `2. models.tex`: Sec. 2.3 末尾に注釈追加 (3.1)
2. `3. estimation.tex`: Algorithm 1 を新 step 2b 込みに更新 (3.2)
3. `3. estimation.tex`: 新サブセクション ``Between level random intercept extraction'' を挿入 (3.3)
4. `3. estimation.tex`: Kalman filter 諸式の augmented 表記を簡素化 (3.4)
5. `A3. estimation.tex`: 三分割構成へ書き換え (3.5)
6. `A1. notation.tex`: 新シンボル $\pi_{is}, \bar{\bm{y}}_{1i\cdot}, \tilde{\bm{y}}_{1it}, \bm{A}_i^{(\pi)}$ を追加 (3.6)
7. `6. discussion.tex`: Open questions に Kalman EM 段落追加 (3.7)
8. `*.bib`: Muthen1994, SnijdersBosker2012, ShumwayStoffer1982 を追加

シミュレーションと実証分析の数値結果は変えない (現行コードは旧 augmented
実装に基づくため)。改訂版の文章中では:

> ``While the current numerical results were produced using the augmented
> state implementation of \citet[Sec.~A3.1]{paper}, the two stage
> procedure described in Section~\ref{sec:estimation} represents the
> principled formulation that we recommend for future applications and
> for a fully integrated likelihood implementation.''

という transparent な notice を一文だけ Section 5 か 6 に入れる。

---

## 7. リスクと未解決点

- **Stage A の周辺尤度導出に regime 平均 $\pi_{is}$ を使う近似**:
  ``individual i が長期的に regime s に滞在する割合'' を sufficient
  statistics に丸めているため、状態確率の不確実性が無視される。
  Reviewer から再度問われた場合は ``Kalman EM が将来研究'' で押さえる。

- **stationarity 前提**: $(I - B_{3is})^{-1}$ の存在を仮定。$|(\bm{B}_{3is})_{jj}|<1$
  という Section 4 の制約と整合。

- **`\pi_{is}` の依存ループ**: regime probability が Kim filter の output
  なので、step 2b は Kim filter の output に依存する。これを
  ``previous iteration の smoothed probabilities'' で打ち切る形にすることで
  循環依存を回避 (Section 3.3 の改訂テキストに明記済)。

- **Tino の合意確認**: REML analog で押すかどうか、彼の好む表現 (Snijders
  Bosker の文言) を尊重。``REML analogous two step'' vs ``MUML inspired
  two step'' のどちらを主軸にするかは meeting で確定済 (REML analog).

---

## 8. ToDo チェックリスト (overleaf 編集用)

- [ ] `2. models.tex` ζ_{2i} について Sec. 2.3 末尾コメント追加
- [ ] `3. estimation.tex` Algorithm 1 改訂
- [ ] `3. estimation.tex` 新 Sub ``Between level random intercept extraction''
- [ ] `3. estimation.tex` Kalman filter 諸式の縮約
- [ ] `A3. estimation.tex` 三分割構成へ
- [ ] `A1. notation.tex` 新シンボル追加
- [ ] `6. discussion.tex` Kalman EM 段落追加
- [ ] `*.bib` 文献追加
- [ ] Section 5 or 6 に ``transparent notice'' 一文
- [ ] response letter R2-2a 本文 (DONE 済) と表記整合チェック

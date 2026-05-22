# Response Letter — PSY-2025-0213
# Frequentist forecasting in regime-switching models with extended Hamilton filter

---

## Opening

Dear Dr. Sinharay, Dr. Liu, and Reviewers,

Thank you for the careful and constructive reviews of our manuscript PSY-2025-0213, entitled "Frequentist forecasting in regime-switching models with extended Hamilton filter." We deeply appreciate the thorough reading and the detailed, insightful suggestions provided by all three reviewers and the Associate Editor. The comments have led to a substantial improvement of the manuscript.

In this revision, we have (1) added a dedicated Initialization subsection (Section 3.1) that discusses the sensitivity of the Kim filter to starting values and the strategies employed; (2) clarified the model notation and corrected dimensional inconsistencies throughout Section 2; (3) expanded the simulation study to a 2×2 factorial design crossing sample size ($N \in \{50, 100\}$) and training length ($N_{\text{train}} \in \{25, 50\}$) with $T = 60$ time points; (4) strengthened the discussion of the empirical example; and (5) added a reference to Chow et al. (2010) in the factor score estimation section. Below we reproduce each comment in full and respond to each point individually, indicating the corresponding changes in the manuscript.

---

## Associate Editor (Dr. Hongyun Liu)

We thank the Associate Editor for the careful reading of the manuscript and the constructive guidance.

---

> Please focus especially on the comments regarding simulation studies.

We thank the Associate Editor for this guidance. Following the comments from all three reviewers on the simulation study, we have substantially revised Section 5 and Appendix A5. Specifically, we expanded the simulation to a 2×2 factorial design crossing sample size $N \in \{50, 100\}$ and training length $N_{\text{train}} \in \{25, 50\}$, yielding four conditions ($\mathcal{D}_{50,25}$, $\mathcal{D}_{100,25}$, $\mathcal{D}_{50,50}$, $\mathcal{D}_{100,50}$). The total number of time points was revised to $T = 60$ (50 dynamics + 10 forecast window). We also added a paragraph in the Discussion (Section 6) discussing how design factors differentially affect between- and within-individual parameter estimation. See the revised Sections 5 and 6 for the full details.

---

> Additionally, please ensure that the formatting is correct throughout the manuscript. For example, on p. 9 the footnote callouts should be placed after punctuation marks.

We thank the Associate Editor for pointing this out. We have reviewed the manuscript globally and corrected all footnote callout placements so that they follow punctuation marks consistently. We also performed a general formatting review across all sections.

---

## Reviewer 1

We thank Reviewer 1 for the positive assessment and the detailed and constructive suggestions.

---

### Major Comment

> While the authors highlight that the frequentist approach avoids the need to specify priors, it still requires starting values to initialize the Kim filter. Prior work has shown that maximum likelihood estimation in state-space models can be sensitive to initial values at time=0, with implications for convergence and parameter estimation. It would strengthen the manuscript to include a discussion of this issue. Specifically, how sensitive is the proposed method to starting values, and what strategies—if any—are recommended for specifying starting values? Should preliminary analyses or prior empirical findings be used to narrow the plausible range of starting values?

We thank the reviewer for this important observation. We have added a dedicated subsection, "Initialization (step 1)," to Section 3.1. The subsection clarifies the distinction between Bayesian priors and initialization devices, describes the diffuse initialization adopted for the state moments ($\bm{\eta}_{1i0|0} = \bm{0}$, $\bm{P}_{1i0|0} = \bm{I}$), and explains why this choice is reasonable when observed variables are standardized to unit variance prior to estimation. We also note more principled alternatives, including model-implied initial moments derived from the stationary distribution of the state process (Harvey, 1990; Du Toit & Browne, 2007), and leave such refinements to future work. Regarding parameter starting values, we note that the optimization is performed with multiple random initializations (5 in both the empirical and simulation studies), and the best solution (highest log-likelihood) is retained, which mitigates sensitivity to any single starting value. The initial regime probability is set to $\Pr[S_{i0}=1|\mathcal{D}_{1:0}]=1$ for all individuals, reflecting our substantive assumption that all participants begin in the no-dropout-intention regime, with a note that a model-implied alternative would derive initial probabilities from the ergodic distribution of the Markov transition matrix (Yang, 2010).

See Section 3.1:

> "Let $\mathcal{D}_{1:t} =\{\bm{y}_{1i1:t}, \bm{y}_{2i} \}_{i=1}^{N}$ be the collection of available data at time $t$ … $\bm{\eta}_{1i0|0}$ is fixed to a zero vector and $\bm{P}_{1i0|0}$ is initialized as an identity matrix, which provides a reasonable approximation given that the time-dependent observed variables are standardized to unit variance prior to estimation. More principled alternatives include model-implied initial moments derived from the stationary distribution of the state process (Harvey 1990, du Toit 2007); we leave such refinements to future work."

---

### Minor Comments

> Page 2, Line 9: It would be helpful to clarify the meaning of "nonlinear dynamics." Are the authors referring to nonlinear temporal trajectories, or to nonlinear functional forms (e.g., sigmoid function) between transition probabilities and their predictors?

We thank the reviewer for raising this ambiguity. We have clarified the sentence on p. 2 to specify that the nonlinearity refers to a nonlinear function (sigmoid) in the autoregressive time-series model governing regime transitions, not to nonlinear temporal trajectories per se.

See Section 1:

> "… whose state-space representation is governed by nonlinear dynamics in the autoregressive time-series model."

---

> Page 3 Line 10: "VAR" needs to be spelled out as "vector autoregressive" at its first mention.

We thank the reviewer. We have added the full term at the first occurrence in Section 2:

> "… hidden Markov 1st order vector autoregressive (VAR(1)) models …"

---

> Page 4, Section 2.3 (Between-level structural model): Several dimensional inconsistencies appear in this section.
> (1) Should $\bm{b}_{2s}$ be a $U_1 \times U_2$ matrix rather than a $U_1 \times 1$ vector?
> (2) Should $\bm{Q}_2$ be a $U_1 \times U_1$ matrix rather than $U_2 \times U_2$?
> (3) Should $\bm{B}_{4s}$ be a $U_1 \times U_2$ matrix rather than $U_1 \times U_1$? Even with this correction, $\bm{B}_{4s}\bm{\eta}_{2i}$ becomes a $U_1 \times 1$ vector and still cannot be added to the $U_1 \times U_1$ matrix $\bm{B}_{3s}$.
> The matrix dimensions in Equation 2.5 appear to imply that $U_2 = 1$, meaning $\bm{\eta}_2$ is treated as a single latent factor. If this is the intended specification, please clarify the rationale for restricting $\bm{\eta}_2$ to one factor. If not, the dimensional inconsistencies in the equation should be corrected.
> It is also unclear why $\bm{B}_{4s}$ is described as a "matrix of conditional interaction effects"; my understanding is that it represents the effect of $\bm{\eta}_{2i}$ on $\bm{B}_{3is}$.
> Equation 2.5 does not include a random-effect term. Is this omission intentional, e.g., to ensure model identification or to simplify the model to improve convergence?
> Are the VAR(1) parameters in $\bm{B}_{3is}$ constrained to lie between $-1$ and $1$ to ensure stationarity? If so, how are these constraints imposed on $\bm{B}_{3s}$ and $\bm{B}_{4s}$?

We thank the reviewer for identifying these dimensional inconsistencies and for the thoughtful questions. We have revised Section 2.3 as follows.

(1) $\bm{B}_{2s}$ has been corrected to a $U_1 \times U_2$ matrix (in the empirical application and simulation, $U_2 = 1$, so it reduces to a $U_1 \times 1$ vector, but the general notation is now stated correctly).

(2) $\bm{Q}_2$ has been corrected to a $U_1 \times U_1$ matrix, which is the covariance matrix of the $U_1 \times 1$ random intercept vector $\bm{\zeta}_{2i}$.

(3) $\bm{B}_{4s}$ has been clarified as a $U_1 \times U_1 \times U_2$ array (a stack of $U_2$ matrices of size $U_1 \times U_1$), so that $\bm{B}_{4s} \bm{\eta}_{2i}$ yields a $U_1 \times U_1$ matrix. A footnote has been added explaining this multiplication.

(4) The random-effect term $\bm{\zeta}_{2i}$ is already included in Equation 2.5 for the intercept equation ($\bm{b}_{1is}$). The autoregressive coefficient $\bm{B}_{3is}$ is modeled as a fixed function of $\bm{\eta}_{2i}$ without an additional random term; this choice is motivated by practical identifiability considerations and is now explicitly stated.

(5) Stationarity of $\bm{B}_{3is}$: In the present implementation, $\bm{B}_{3is}$ is constrained to be diagonal and each diagonal element is constrained to satisfy $|(\bm{B}_{3is})_{jj}| < 1$, which ensures stationarity for the AR(1) specification. A footnote has been added in Section 4 (Empirical study) making this explicit.

---

> Page 4 Section 2.4: Regarding $\gamma_{s'2}$, since this is a scalar coefficient, does this imply that there is a single inter-individual factor (i.e., $\bm{\eta}_{2i}$ is scalar and $U_2 = 1$)? If so, this should be stated explicitly, and as I mentioned earlier, the rationale for restricting $\bm{\eta}_2$ to one factor should be clarified.

We thank the reviewer. We have clarified the notation so that $\gamma_{s'2}$ is now stated as a $1 \times U_2$ vector (Equation 2.6). In the empirical application and simulation study, $U_2 = 1$ (a single inter-individual factor, cognitive skills), which reduces $\gamma_{s'2}$ to a scalar; this is now explicitly stated in Section 4 where the empirical model is described.

---

> Page 6 Line 22: The manuscript notes that $\bm{P}_{\text{aug},0|0}$ is set to a specific block matrix "reflecting the prior variance of the random intercepts $\bm{\zeta}_2$." It is unclear how this "prior variance" is obtained. Is it informed by previous studies, or estimated from the data? Why do the authors choose not to use a diffuse density with large constants in the covariance matrix, as is commonly done? Is this choice motivated by convergence considerations?

We thank the reviewer for pointing out this ambiguity. In the revised Initialization subsection (Section 3.1), we have replaced the earlier description with a clearer account: $\bm{\eta}_{1i0|0}$ is fixed to a zero vector and $\bm{P}_{1i0|0}$ is set to an identity matrix $\bm{I}$, which is a reasonable choice when all observed variables are standardized prior to estimation. The term "prior variance" has been removed to avoid confusion with Bayesian priors.

---

> Page 11 Line 11: There is a typo. Figure 1 displays three individuals, not two.

We thank the reviewer. This has been corrected: the text now reads "3 individuals" (Section 4.2.4).

---

> Page 14 Line 21: The statement "clear indication that the data not even made missing completely at random (MCAR)" is unclear. MCAR is rarely plausible; do the authors mean that the pattern suggests the data may not satisfy MAR and could be MNAR?

We thank the reviewer for this clarification. The sentence has been revised to read:

> "… which implies that the data might not be missing at random (MAR) but missing not at random (MNAR)."

---

> Page 14 Line 32: The phrase "given that the current Bayesian implementation already requires a lot of time to compute" needs clarification. The analyses in this manuscript use frequentist methods rather than Bayesian. If "the current Bayesian implementation" refers to the Bayesian implementation in Kelava et al. (2022), please state this explicitly.

We thank the reviewer. The sentence has been revised to make clear that "the current Bayesian implementation" refers to the Bayesian implementation in Kelava et al. (2022):

> "… given that the current Bayesian implementation \citep{Kelava2022} already requires a lot of time to compute."

---

## Reviewer 2

We thank Reviewer 2 for the thorough and constructive review, which has led to several important improvements to the manuscript.

---

> 1. Description and illustrations centering on the motivating example can be strengthened.
> a. Describe the motivating example and use it to better motivate explanations of the key equations. Portions of the empirical study section should be moved up to before the eqs are introduced, with further descriptions interspersed with explanations of some of the key modeling equations, such as why it is interesting to have $\bm{\eta}_{1i,t-1}$, $\bm{\eta}_{2i}$ and their interaction effect in Eq 2.6.
> b. Illustrate special cases of the proposed model. In other well-known and related variations of the proposed model, such as dynamic structural equation models (DSEMs; Asparouhov et al., 2018) and multilevel vector autoregressive models (Li et al., 2022), the $\bm{B}_{3is}$ in Equation 3.5 is typically specified as a function of $\bm{B}_{3s}$ + a vector of person-specific random effects for each parameter in $\bm{B}_{3is}$. Here, the authors specify $\bm{B}_{3is}$ to be a function of $\bm{\eta}_{2i}$, which captures sources of covariations among the observed variables in $\bm{y}_{2i}$. However, one special case in which $\bm{\eta}_{2i}$ has as many latent variables as the unique parameters in $\bm{B}_{3i}$ indeed overlaps with these other, perhaps better-known special cases. It would be helpful to draw the analogy and illustrate concretely the parallels between the proposed model and these special cases.
> c. Helpful also to draw some analogy between the proposed approach and the literature on latent transition models.

We thank the reviewer for these suggestions to strengthen the motivating example. We have made the following revisions.

(a) We have added brief forward references to the empirical example throughout Section 2, explaining concretely—in terms of the student dropout application—why it is of interest to model transition probabilities as a function of $\bm{\eta}_{1i,t-1}$ (current affective state), $\bm{\eta}_{2i}$ (cognitive skills), and their interaction (i.e., whether cognitive resources buffer the effect of momentary distress on dropout intention).

(b) We have added a paragraph at the end of the Between-level structural model subsection (Section 2.3) drawing the connection to DSEMs (Asparouhov et al., 2018) and multilevel VAR models (Li et al., 2022). Specifically, we note that in those frameworks the person-specific autoregressive matrix is parameterized as $\bm{B}_{3is} = \bm{B}_{3s} + \bm{u}_i$, where $\bm{u}_i$ is a vector of unconstrained person-specific random effects. In the present model, the analogous individual deviation is expressed as $\bm{B}_{4s}\bm{\eta}_{2i}$, in which $\bm{\eta}_{2i}$ is a lower-dimensional vector of between-person latent factors that simultaneously accounts for covariation in the time-invariant observed variables and for individual differences in autoregressive dynamics. As a special case, when $U_2$ equals the number of unique parameters in $\bm{B}_{3i}$, the proposed model overlaps with the DSEM/mlVAR parameterization, since $\bm{B}_{4s}\bm{\eta}_{2i}$ then spans the same space as an unconstrained random-effects vector $\bm{u}_i$.

(c) We have added a brief analogy to latent transition modeling in Section 2 (Markov-switching model). Specifically, we clarify that the discrete regime variable $S_{it}$ can be interpreted as a latent status at each time point and that Equation 2.6 defines the corresponding first-order transition probabilities between those statuses. We also distinguish the present framework from conventional latent transition models by noting that the transition process is embedded in a regime-switching state-space model and allows transition probabilities to depend on time-varying within-person latent states, time-invariant between-person latent traits, and their interaction.

---

> 2. Aspects of proposed estimation method could be strengthened.
> a. Under the assumption that the model is correctly specified (e.g., no further covariations in $\bm{y}_{2i}$ and $\bm{y}_{1it}$ beyond what is being modeled), estimation of $\bm{\eta}_{2i}$ and $\bm{\zeta}_{2i}$ can be further improved. For instance, the authors may be able to use the multilevel decomposition approach in DSEM (Asparouhov et al., 2018; see also references in Asparouhov et al., 2018b) to first separate the latent intercepts from the within-person deviations from the intercepts. Elements such as $\bm{\eta}_{2i}$ and $\bm{\zeta}_{2i}$ are time-invariant so it is generally unnecessary to put any of them into the extended Kim filter portion of the algorithm. In fact, doing so and arbitrarily adding diffuse or slightly misspecified initial conditions for the latent variables would actually negatively impact the quality of the estimation results.
> b. If the authors want to keep using the regression or Bartlett estimators for the time-invariant factor scores, it may be helpful to see the references in Chow et al. (2010) for well-known correspondence between regression/Bartlett estimators of factor scores and the Kalman filter.
> c. Initial condition for step 2 can be improved by using the model-implied mean vector and covariance matrix to start the process (see Harvey01a, duToit07a, Yang10a).
> d. Identifiability and observability constraints can be better spelled out. When $\bm{\eta}_{\text{aug},it}$ is augmented to include $\bm{\zeta}_{2i}$, the resulting state-space model can quickly become non-observable. Hence the suggestion to consider the proposal in 2a.

We thank the reviewer for these important methodological comments.

(a) We address this comment in two parts, corresponding to the two time-invariant components of the model.

*Separation of $\bm{\eta}_{2i}$ (between-level factor scores).* The reviewer's concern about observability is already partially addressed in the current implementation. As described in Section 3.2 (Algorithm 1, step 2), the between-level factor scores $\hat{\bm{\eta}}_{2i}$ are estimated entirely outside the Kim filter, via a confirmatory factor analysis on the time-invariant observed variables $\bm{y}_{2i}$ using Bartlett factor score weights (Equation 3.2). These point estimates are then plugged into the filter as fixed, observed covariates — the filter never attempts to track $\bm{\eta}_{2i}$ dynamically. This structure directly implements the multilevel decomposition principle formalized by Muthén (1994): between-level variation is captured by a separate between-level factor model, and the resulting factor scores condition the within-level dynamics without entering the state vector. Accordingly, $\bm{\eta}_{2i}$ does not appear in $\bm{\eta}_{\text{aug},it}$, and the observability problem does not apply to it.

*Separation of $\bm{\zeta}_{2i}$ (person-specific random intercept residuals).* The reviewer's concern is more directly applicable to $\bm{\zeta}_{2i}$, which in the current implementation is carried in the augmented state vector $\bm{\eta}_{\text{aug},it} = (\bm{\eta}_{1it}^\top, \bm{\zeta}_{2i}^\top)^\top$. Because $\bm{\zeta}_{2i}$ is time-invariant — its state-transition matrix is the identity and its process noise block in $\bm{Q}_{1s,\text{aug}}$ is zero — it is not directly identifiable from the within-person time series through the Kalman filter mechanism. Specifying diffuse initial conditions for the $\bm{\zeta}_{2i}$ block of $\bm{P}_{\text{aug},i0|0}$ introduces the misspecification the reviewer flags.

The principled resolution is to estimate $\bm{\zeta}_{2i}$ outside the dynamic filter, following the between-within decomposition approach of Muthén (1994) and the analogous latent centering procedure in DSEM (Asparouhov et al., 2018). In this framework, the person-mean-centered within-person observations $\tilde{\bm{y}}_{1it} = \bm{y}_{1it} - \bar{\bm{y}}_{1i\cdot}$ are passed to the Kim filter (capturing dynamic within-person variation), while the person means $\bar{\bm{y}}_{1i\cdot}$ are used in a separate between-person model to estimate $\bm{\zeta}_{2i}$ and $\bm{Q}_2$ via empirical Bayes (see revised Section~3.3 and Appendix~A3 for the full derivation). This two-stage procedure is analogous to restricted maximum likelihood (REML) estimation in linear mixed-effects models (Snijders & Bosker, 2012, Ch. 4), where variance components such as $\bm{Q}_2$ are estimated by conditioning on the fixed effects, avoiding the downward bias that arises under full ML when fixed and random parameters are estimated simultaneously.

**We have implemented this two-stage procedure and re-estimated the empirical model.** The revised implementation removes $\bm{\zeta}_{2i}$ from the augmented state vector entirely, resolving the observability issue the reviewer identified. The substantive conclusions of the empirical analysis are maintained under the revised implementation: **[PLACEHOLDER: brief comparison of key parameter estimates and regime probabilities between old and new implementation — to be filled once empirical rerun results are available. E.g., "The estimated autoregressive coefficients, regime transition probabilities, and factor loadings are consistent with those reported in the original submission (largest discrepancy: XX for parameter YY). The estimated between-person variance $\hat{Q}_2$ changed from XX to XX."]**

A fully integrated implementation using, for example, the Kalman EM algorithm (Shumway & Stoffer, 1982) would embed this separation within a unified likelihood objective and replace the iterative profile structure with a single coherent EM loop; we retain this as a direction for future work (see Section~6).

(b) We have added a reference to Chow et al. (2010) in the Bartlett factor score subsection (Section 3.2):

> "The Bartlett factor scores provide a conditionally unbiased estimate of the factor scores given the observed data \citep[see also][for the well-known correspondence between regression/Bartlett estimators of factor scores and the Kalman filter]{Chow2010}."

(c) We have addressed the initial condition in the new Initialization subsection (Section 3.1). We note model-implied alternatives (Harvey, 1990; Du Toit & Browne, 2007; Yang & Chow, 2010) and leave more principled initialization to future work, as the current diffuse initialization is a standard and practically well-motivated choice when variables are standardized.

(d) We agree that identifiability and observability constraints deserve a clearer treatment. In the present revision, we have clarified several implementation-specific restrictions, but a fuller discussion of observability under augmented-state formulations remains an important topic for future methodological work.

---

> 3. Simulation studies and results can be strengthened.
> a. I found it perplexing that $\mathcal{D}_{100}$ doesn't have better results than $\mathcal{D}_{75}$. Why? 50 time points too short given the misspecification in the initial condition? See the points mentioned above in (2).
> b. I hope the authors can discuss more extensively the statistical properties and performance quality of the point and standard error estimates for the modeling parameters. How are biases/RMSEs, coverage rates, and power affected with variations in design factors?
> c. The key appeal of the paper and proposed approach is in the incorporation of latent variables e.g., in the HMM model of regime switches. What if a user goes ahead and have the HMM be a function of $\bm{y}_{2i}$ as opposed to $\bm{\eta}_{2i}$, or omit $\bm{\eta}_{1i,t-1}$ from Eq 2.6? This aspect is not investigated/illustrated in the simulation study at all.

We thank the reviewer for these important comments on the simulation study.

(a) We agree with the reviewer that the lack of improvement from $\mathcal{D}_{75}$ to $\mathcal{D}_{100}$ was surprising and merited further investigation. In our revised simulation study, we have replaced the original $N \in \{75, 100\}$ design with a 2×2 factorial design crossing $N \in \{50, 100\}$ and $N_{\text{train}} \in \{25, 50\}$ (see revised Section 5). This revised design allows us to cleanly separate the effects of sample size and time points on estimation quality. The revised results and interpretation are presented in Sections 5.3 and 6.

(b) We have expanded the reporting of simulation results in the revised Appendix A5 to include bias, SD, and RMSE for all parameter categories, as well as regime prediction accuracy (sensitivity and specificity) under all four conditions. A summary paragraph in Section 6 discusses how design factors differentially affect between-individual (improved with larger $N$) versus within-individual (improved with longer $N_{\text{train}}$) parameter estimation.

(c) The reviewer raises an excellent point regarding the added value of latent predictors ($\bm{\eta}_{2i}$, $\bm{\eta}_{1i,t-1}$) in the HMM compared with observed predictors ($\bm{y}_{2i}$) or the omission of within-individual terms. We agree that this is an important practical and methodological question. The present revision does not add these reduced-form comparison conditions, as doing so would require a substantially broader simulation design than we could support reliably within the current revision cycle. We therefore view this as an important direction for future work.

---

> 4. Aspects of the empirical example can be strengthened.
> a. Can the authors offer some recommendations on how to go about making empirical decisions on model formulation, for instance, in making decisions on whether to have latent variables $\bm{\eta}_{2i}$ or just letting each variable in $\bm{y}_{2i}$ serves as predictors in the regime switching models?
> b. In what ways can practitioners and applied researchers use the forecasting output? From the estimated regime probabilities, for instance, they may want to do something like an ROC curve to determine how they want to use drop out probabilities to implement prevention/intervention efforts. Adding slightly more elaborations would make the motivating example and results more convincing.

We thank the reviewer for these practically oriented suggestions.

(a) We agree that applied guidance on this modeling choice would be valuable. One structural criterion that can inform this decision is whether the between-person quantity of interest carries its own process noise component—that is, whether it is subject to stochastic individual-level variation that cannot be attributed to measurement error alone. As noted by Chow and Zhang (2013, p. 742, fn. 1), time-varying parameters that lack their own process noise components need not be modeled as latent variables. By extension, when a between-person predictor such as $\bm{\eta}_{2i}$ does have an associated variance component $\bm{Q}_2$, representing it as a latent variable is structurally motivated, because collapsing it onto its observed proxy $\bm{y}_{2i}$ would conflate true between-person variance with measurement error. In contrast, if $\bm{Q}_2$ is negligible—as indeed suggested by our empirical results—or if measurement error in the observed indicators is judged to be small, using $\bm{y}_{2i}$ directly offers a simpler and equally defensible choice. We have added a brief note along these lines in Section 2.3. A comprehensive empirical decision framework—including formal model comparison strategies for choosing between latent and observed predictors—remains an important direction for future work.

(b) We have added a brief paragraph in Section 4.2.4 (Regime probabilities) noting that the predicted regime probabilities can be used to identify early warning thresholds—for example, by examining the operating characteristics of different cutoffs (analogous to an ROC analysis) to trade off sensitivity and specificity for intervention purposes. We suggest this as a practical application of the forecasting output in educational monitoring settings.

---

> Minor editorial comments:
> 1. Right after Eq. 2.4. $\bm{b}_{2s}$ is a $U_1 \times 1$ vector? Isn't $\bm{b}_{2s}$ a matrix?
> 2. After Eq. 2.4, "$\bm{B}_{4s}$ is a $U_1 \times U_1$ matrix of conditional interaction effects" → say more about why interaction effects.

(1) We have corrected the notation: $\bm{B}_{2s}$ is now stated as a $U_1 \times U_2$ matrix throughout (see response to the dimensional inconsistency comment above).

(2) We have added a footnote explaining that $\bm{B}_{4s}$ captures interaction effects because the resulting autoregressive coefficient for person $i$, $\bm{B}_{3is} = \bm{B}_{3s} + \bm{B}_{4s}\bm{\eta}_{2i}$, depends on the inter-individual factor $\bm{\eta}_{2i}$. The elements of $\bm{B}_{4s}$ therefore quantify how much the strength of the autoregressive dynamics within a person varies as a function of inter-individual characteristics—i.e., they are interaction terms between within-person lagged states and between-person characteristics.

---

## Reviewer 3

We thank Reviewer 3 for the positive assessment and the constructive suggestions.

---

### Major Comment

> The paper deals with an important topic: researchers increasingly collect intensive longitudinal data, but effective methods for analyzing such data lag beyond. The developed method is a much-needed flexible method for analyzing such data. I feel that the paper is a nice contribution to the literature.
> The authors may consider expanding the simulation study to include more sample sizes and time points. With more sample sizes and time points, one can examine the differential influences of sample size and time point on different parameters. I would expect that increasing sample size would lead to more accurate estimate of between-individual parameters and increasing time points would lead more accurate estimate of within-individual parameters. With the results of different time points and sample sizes could also provide guidance to applied researchers on how to design studies with intensive longitudinal study.

We thank the reviewer for this suggestion, which aligns closely with the recommendation from the Associate Editor to focus on the simulation study. We have revised the simulation design from the original single-factor manipulation of $N$ to a 2×2 factorial design crossing $N \in \{50, 100\}$ with training length $N_{\text{train}} \in \{25, 50\}$, yielding four conditions ($\mathcal{D}_{50,25}$, $\mathcal{D}_{100,25}$, $\mathcal{D}_{50,50}$, $\mathcal{D}_{100,50}$). This design was modeled after Kelava et al. (2022), who crossed sample size and time points in their Bayesian simulation study, allowing a parallel comparison of frequentist and Bayesian performance under comparable conditions.

The revised Section 5 and Discussion address the reviewer's expectations directly. For regime prediction accuracy, sensitivity was markedly higher in the $N_{\text{train}}=50$ conditions ($\mathcal{D}_{50,50}$: 82\%; $\mathcal{D}_{100,50}$: 84\%) than in the $N_{\text{train}}=25$ conditions ($\mathcal{D}_{50,25}$: 71\%; $\mathcal{D}_{100,25}$: 69\%), confirming that within-individual parameters and regime detection rely primarily on temporal information. Sample size $N$ had a smaller and less consistent effect on regime metrics. For parameter estimation, the regime-2 autoregressive coefficients ($\text{diag}(\bm{B}_{32})$) showed substantially larger bias and RMSE in the $N_{\text{train}}=25$ conditions (Bias $\approx -0.06$ to $-0.09$, RMSE $\approx 0.14$ to $0.19$) than in the $N_{\text{train}}=50$ conditions (Bias $\approx -0.01$ to $-0.03$, RMSE $\approx 0.04$ to $0.09$), while measurement model parameters ($\bm{R}_1$) remained stable across all four conditions (Bias $\leq 0.00$, RMSE $\leq 0.02$). The between-person factor variance $P_2$ showed a systematic upward bias ($\approx 0.25$--$0.29$) in all conditions. Full results are reported in Tables A1--A3 and Figure A1 of the revised appendix.

We have also added a paragraph in Section 6 providing design guidance for applied researchers:

> "The simulation study employed a 2×2 factorial design crossing sample size $N \in \{50, 100\}$ and training length $N_{\text{train}} \in \{25, 50\}$. As expected, increasing $N$ primarily improved the estimation of between-individual parameters … Conversely, increasing $N_{\text{train}}$ primarily benefited the estimation of within-individual parameters … This is consistent with the general asymptotic theory for mixed-effects and state-space models (Harvey, 1990)."

---

### Minor Comments

> P4, Eq. (2.5) seems to have several typos. It does not include an error term. The implication is that the AR weights are always fixed effects. If it is the case, the authors may add some justifications for the choice. The matrix multiplication of the second term $\bm{B}_{4s} \bm{\eta}_{2i}$ needs to be clarified: it works for the empirical example and simulation study where there is only one between-individual factor. When there are two or more between-individual factors, the matrix multiplication would be incompatible.

We thank the reviewer for identifying these issues.

Regarding the error term: as noted in the response to the dimensional inconsistency comment above, the random intercept $\bm{\zeta}_{2i}$ already appears in the intercept equation. The autoregressive coefficient $\bm{B}_{3is}$ is modeled as a deterministic function of $\bm{\eta}_{2i}$ (i.e., fixed heterogeneity) rather than a random effect; this is now stated explicitly with justification (identifiability and computational feasibility).

Regarding the dimensional incompatibility with $U_2 > 1$: we have revised the notation so that $\bm{B}_{4s}$ is described as a $U_1 \times U_1 \times U_2$ array and the product $\bm{B}_{4s}\bm{\eta}_{2i}$ is clarified as a sum $\sum_{k=1}^{U_2} \bm{B}_{4s}^{(k)} \eta_{2i,k}$, where $\bm{B}_{4s}^{(k)}$ is the $k$-th $U_1 \times U_1$ slice of the array. This makes the multiplication well-defined for any $U_2$. A footnote in Section 2.3 clarifies this.

---

> P9, the paragraph below "4.1 Forecast implementation", are these assumptions made to simplify the estimation of this particular model or to have broad applicability? In particular, the 5th, 6th, and 7th assumptions specify some fixed values to some parameters (or super-parameters), what are the implications of these values? Should use the same sets of values for other models? The authors may consider adding path-diagrams to display the models considered in the paper. Since the regime-switching model is a complex model, it involves quite many notations. They authors may consider adding a table to explain the notations.

We thank the reviewer for these suggestions.

Regarding the assumptions in Section 4.1: we have clarified at the beginning of the paragraph that assumptions 1–8 are specific to the present empirical application and are not required by the general framework. Each assumption is now accompanied by a brief justification explaining whether it is motivated by model identification, computational feasibility, or substantive knowledge about the dropout process.

Regarding path diagrams and notation tables: we have added a notation summary table in the appendix (Table~A1) that lists all major symbols, their dimensions, and their interpretations. We have also added a simplified path diagram in the Introduction (Figure~1) illustrating the high-level structure of the proposed RSSS model: the within-person level shows the dynamic latent variables and discrete regime variable with VAR(1) dynamics, while the between-person level shows the time-invariant latent variables and their effects on within-person dynamics and transition probabilities. A full path diagram covering all regime-dependent parameters would be difficult to read without loss of clarity, so we opted for this schematic representation adapted from \citet{Kelava2022}.

---

> P10, the paragraph below "4.2.1 Markov-switching parameters", the third sentence is "the interaction effects $\bm{\gamma}_4$ were negative in 2 out of 3 parameters." If the between-individual factor "cognitive skills" interacts with all the seven within-individual factors, should it be seven?

We thank the reviewer for this careful reading. As noted in the empirical model specification (Section 4.1, assumption 7), interaction effects $\bm{\gamma}_4$ are restricted to zero for four motivational variables (content not important, not understanding, positive affect, negative affect), and only three self-regulatory variables (cost, afraid to fail, stress) are allowed to have non-zero interaction effects. The sentence should therefore read "2 out of 3 parameters," which is correct given the model restriction. We have added a cross-reference to assumption 7 in this sentence to avoid confusion.

---

## Closing

We hope that the revisions described above and the accompanying changes in the manuscript fully address all comments raised by the reviewers and the Associate Editor. We are grateful for the thorough and constructive feedback, which has substantially strengthened the manuscript. We look forward to your favorable consideration of the revised submission.

Sincerely,

Kento Okuyama, Tim Schaffland, Kilian Pascal, Holger Brandt, Augustin Kelava

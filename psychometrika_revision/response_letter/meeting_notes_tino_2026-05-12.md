# Meeting Notes — Revision Discussion with Tino
**Date:** 2026-05-12  
**Paper:** PSY-2025-0213 "Frequentist forecasting in regime-switching models with extended Hamilton filter"  
**Deadline:** June 9, 2026 (major revision, Psychometrika)

---

## Color coding system for revision tracking

Tino's convention for tracking revision status in the response letter and manuscript:

- **Blue** = solved / addressed
- **Red** = not yet done / needs attention

---

## Status at time of meeting

Most revision items are complete (marked blue in manuscript with `\textcolor{red}{...}` = new additions). The one remaining RED item is:

**Reviewer 2, comment 2a: Between-within separation of random effects**

---

## R2-2a: Between-within separation — key discussion points

### The problem

Reviewer 2 pointed out that the current implementation places both $\eta_{2i}$ (between-level factor scores) and $\zeta_{2i}$ (person-specific random intercept residuals) in the augmented Kim filter state vector. Because these are time-invariant, they are not directly observable through the Kalman filter mechanism. Using diffuse initial conditions for these components introduces misspecification.

### What is already done

$\eta_{2i}$ is **already estimated outside the Kim filter** via Bartlett factor scores from $\bm{y}_{2i}$ (Algorithm 1, step 2 / Section 3.2). This correctly follows the Muthén (1994) multilevel decomposition principle: between-level variation is captured by a separate between-level factor model, and factor scores are passed as fixed covariates into the within-level dynamics.

### What still needs to be done

$\zeta_{2i}$ (person-specific random intercept residuals) is still carried in the augmented state vector. This is the observability problem the reviewer identifies. The correct fix is to estimate $\zeta_{2i}$ separately from the dynamic filter:

1. Pass person-mean-centered observations $\tilde{\bm{y}}_{1it} = \bm{y}_{1it} - \bar{\bm{y}}_{1i\cdot}$ to the Kim filter
2. Use person means $\bar{\bm{y}}_{1i\cdot}$ in a separate between-person model to estimate $\zeta_{2i}$ and $Q_2$

This is analogous to **REML** (Restricted Maximum Likelihood) in linear mixed-effects models — variance components ($Q_2$) are estimated by conditioning on the fixed effects to avoid downward bias from simultaneous ML estimation.

### References Tino provided

- **Muthén, B. (1994).** Multilevel Covariance Structure Analysis. *Sociological Methods & Research*, 22(3), 376–398.
  - Establishes the between-within decomposition: $\Sigma_T = \Sigma_W + \Sigma_B$
  - Between-level covariance matrix $\Sigma_B = \Lambda_B \Psi_B \Lambda_B' + \Theta_B$ estimated from between-group sample covariance matrix $S_B$
  - Within-level $\Sigma_W$ estimated from pooled-within matrix $S_{PW}$
  - FIML ("full information ML") vs. MUML ("Muthén's ML-based estimator")

- **Muthén, L. (2012).** FIML Technical Report (p. 34)
  - Technical treatment of full-information ML in multilevel CFA

- **Snijders, T.A.B. & Bosker, R.J. (2012).** *Multilevel Analysis*, 2nd ed. Sage.
  - Ch. 4: REML for variance component estimation in hierarchical linear models
  - REML separates estimation of fixed effects from random variance components, avoiding ML downward bias
  - Website with materials: https://www.stats.ox.ac.uk/~snijders/mlbook.htm

- **Asparouhov, T., Hamaker, E.L., & Muthén, B. (2018).** Dynamic Structural Equation Models. *SEM*, 25(3), 359–388.
  - DSEM uses "latent centering" to decompose total variance into within-person and between-person parts
  - Time-invariant components ($\eta_{2i}$) estimated outside the time-series filter

- **Kalman EM / Kalman ML**
  - Shumway & Stoffer (1982): EM algorithm for state-space models
  - A fully integrated implementation would embed the $\zeta_{2i}$ separation within a unified EM objective

---

## Action items

1. **Response letter R2-2a**: Rewrite from "future work" to substantive two-part response:
   - Part 1: Clarify $\eta_{2i}$ is already separated (CFA step)
   - Part 2: Acknowledge $\zeta_{2i}$ observability issue, describe REML-analog two-step separation as solution
   - Reference: Muthén (1994), Asparouhov et al. (2018), Snijders & Bosker (2012)
   - *Status: DONE (2026-05-12)*

2. **Manuscript Section 3.3 note**: Add description of $\zeta_{2i}$ limitation and the two-step separation approach

3. **Future work (for next paper)**: Full Kalman EM implementation integrating $\zeta_{2i}$ estimation within the unified likelihood objective

---

## Other revision items (all complete as of this meeting)

| Item | Status |
|---|---|
| Initialization subsection (Section 3.1) | Done |
| Simulation 2×2 factorial design ($N \times N_{\text{train}}$) | Done |
| Notation table (Appendix) | Done |
| Path diagram (Figure 1, Introduction) | Done |
| DSEM/mlVAR analogy (Section 2.3) | Done |
| Motivating example expanded | Done |
| Dimension corrections ($B_{2s}$, $Q_2$, $B_{4s}$) | Done |
| ROC analogy paragraph | Done |
| Footnote formatting | Done |
| R2-2b: Chow et al. (2010) reference added | Done |
| R2-2c: Initialization model-implied alternatives noted | Done |
| R2-2d: Observability mention added | Done |

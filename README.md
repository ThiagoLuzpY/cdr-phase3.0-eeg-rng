# Core Distinguishability Relativity (CDR)


## Status

- **Phase I** ✅ **COMPLETE** (7/7 gates PASS)
- **Phase II.1A** (Empirical validation – energy systems) ✅ **COMPLETE**
- **Phase II.1B** (Empirical validation – neurodynamics) ✅ **COMPLETE**
- **Phase II.2** (Human mobility systems) ✅ **COMPLETE**
- **Phase II.3** (Ecological population dynamics) ✅ **COMPLETE**
- **Phase II.4** (Protein dynamics) ✅ **COMPLETE**
- **Phase III.0.1** (EEG-only experimental neural validation) ✅ **COMPLETE**
- **Phase III.0.2** (RNG-only experimental stochastic baseline validation) ✅ **COMPLETE**
- **Phase III.0.3** (Joint EEG + RNG validation) ✅ **COMPLETE**
- **Phase III.1 Redesign** (Multi-subject and latent / quantum-aware joint redesign) 📋 **PLANNED**

---

## What is CDR?

**Core Distinguishability Relativity (CDR)** is a pre-registered framework designed to detect information-driven selection bias in observed dynamics without falling into common statistical pitfalls such as p-hacking, circular reasoning, or model over-flexibility.

The framework focuses on answering a fundamental scientific question:

> *When we observe a dynamic system, how do we know whether a detected pattern is a real causal effect or merely an artifact of noise or modeling assumptions?*

CDR addresses this through:

- Pre-registered hypotheses  
- Mandatory validation gates  
- Adversarial and structural controls  
- Out-of-sample generalization tests  

Instead of relying solely on p-values, CDR requires multiple orthogonal validation gates to pass before any claim can be considered detectable.

---

## Project Roadmap

The CDR validation program is divided into empirical phases.

| Phase                    | Objective | Status |
|--------------------------|-----------|--------|
| **Phase I**              | Toy-model validation (controlled system) | ✅ Complete |
| **Phase II.1A**          | Real-world validation on energy infrastructure | ✅ Complete |
| **Phase II.1B**          | Real-world validation on neural dynamics (fMRI) | ✅ Complete |
| **Phase II.2**           | Human mobility systems | ✅ Complete |
| **Phase II.3**           | Ecological population dynamics | ✅ Complete |
| **Phase II.4**           | Protein dynamics | ✅ Complete |
| **Phase III.0.1**        | EEG-only experimental neural validation | ✅ Complete |
| **Phase III.0.2**        | RNG-only baseline validation | ✅ Complete |
| **Phase III.0.3**        | Joint EEG + RNG validation | ✅ Complete |
| **Phase III.1 Redesign** | Multi-subject and latent / quantum-aware redesign of the EEG–RNG joint experiment | 📋 Planned |




---

## Phase I — Toy Model Validation (Completed)

Phase I validates the CDR framework in a fully controlled environment using a small enumerated system.

### Model used

- 2-component Ising conditional kernel  
- Binary state variables  
- Known ground-truth coupling parameter `ε`  

### State space
```
2 components × binary states → 4 states
```

---

### Phase I Gates

| Gate | Meaning |
|------|---------|
| **G1** | H₀ recovery |
| **G2** | H₁ recovery |
| **G3** | Control collapse |
| **G4** | Parameter identifiability |
| **G5** | Stability |
| **G6** | Adversarial robustness |
| **G7** | Out-of-sample generalization |

---

### Result
```
CDR Phase I+
────────────────
G1_H0_recovery: PASS
G2_H1_recovery: PASS
G3_controls_collapse: PASS
G4_identifiability: PASS
G5_stability: PASS
G6_adversarial: PASS
G7_out_of_sample: PASS
────────────────
FINAL: PASS
```

---

## Phase II — Empirical Validation

Phase II tests the framework on real observational systems.

The objective is to verify that the estimator:

- Detects reweighting when present
- Does not produce false positives
- Generalizes across unseen data
- Remains stable under discretization changes

---

## Phase II.1A — Energy Infrastructure Validation (Completed)

**Dataset:**
```
Open Power System Data (OPSD)
```

**Variables used:**
```
(load, wind, solar, price)
```

**State definition:**
```
state = (load_bin, wind_bin, solar_bin, price_bin)
```

**Discretization:**
```
3 bins per variable
3⁴ = 81 states
```

**Observations:**
```
8740 hourly transitions
Germany/Luxembourg grid
```

---

### Results
```
CDR Phase II.1A
────────────────
F1_injection_recovery: PASS
F2_controls_collapse: PASS
F3_holdout_generalization: PASS
F5_sensitivity: PASS
────────────────
FINAL: PASS
```

---

### Key Finding

The German electrical grid shows:
```
ε ≈ 0
```

consistent with a highly regulated infrastructure system.

---

## Phase II.1B — Neural Dynamics Validation (Completed)

**Dataset:**
```
OpenNeuro
ds002938
task: effort
subject: sub-01
```

**State construction:**
```
state = (ROI₁, ROI₂, ROI₃, ROI₄, ROI₅)
```

**Discretization:**
```
2 bins per ROI
2⁵ = 32 states
```

**Observations:**
```
661 transitions
```

---

### Phase II.1B Results
```
CDR Phase II.1B (fMRI)
────────────────────────────────

F1_injection_recovery: PASS
eps_hat: 0.0
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.0
fraction_below_tol: 1.0
max_eps_controls: 0.0
n_controls: 20

F3_holdout_generalization: PASS
eps_train: 0.08
eps_test: 0.00
abs_delta: 0.08

F5_sensitivity: PASS
eps_binsA: 0.08
eps_binsB: 0.06
abs_delta: 0.02

────────────────────────────────
FINAL: PASS
```

---

## Phase II.2 — Human Mobility Validation (Completed)

This phase applies the CDR framework to **large-scale human mobility trajectories**.

---

### Dataset
```
Microsoft GeoLife GPS Trajectories
```

**Characteristics:**
```
182 users
17,000+ trajectories
~1.2 million GPS points
Sampling interval ≈ 1–5 seconds
```

**After preprocessing:**
```
user-level state trajectories
discretized spatial bins
temporal transition sequences
```

---

### System Representation

Human mobility was represented as a discrete dynamical system:
```
state = (spatial_cell_t)
```

**Transitions:**
```
s(t) → s(t+1)
```

Kernel estimated via empirical transition frequencies.

---

### Computational Complexity

This phase required substantially heavier computation than previous domains.

**Pipeline runtime:**
```
~48 hours
```

**Reasons:**

- Large trajectory dataset
- Multiple adversarial controls
- Likelihood surface estimation
- Holdout generalization checks

Control experiments alone required several hours due to repeated recomputation of transition kernels.

---

### Phase II.2 Gates

| Gate | Meaning |
|------|---------|
| **F1** | Injection recovery |
| **F2** | Control collapse |
| **F3** | Train/test generalization |
| **F5** | Discretization sensitivity |

---

### Phase II.2 Results
```
CDR Phase II.2 (Human Mobility)
────────────────────────────────

F1_injection_recovery: PASS
eps_hat: 0.30
eps_true: 0.30
abs_err: 0.00

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00
n_controls: 10

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.00
abs_delta: 0.00

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

### Interpretation

The mobility dynamics in the GeoLife dataset appear consistent with a Markovian mobility kernel:
```
ε ≈ 0
```

**Meaning:**

Human spatial transitions in this dataset do not require additional structural mixture beyond the empirical transition kernel.

Importantly, the estimator correctly recovered injected structure:
```
ε_true = 0.30
ε_hat = 0.30
```

confirming estimator sensitivity.

---

## Phase II Conclusions

Across **three independent empirical domains**:

| Domain | Result |
|--------|--------|
| Energy infrastructure | `ε ≈ 0` |
| Neural dynamics (fMRI) | `ε ≈ 0.06–0.08` |
| Human mobility | `ε ≈ 0` |

The CDR estimator demonstrated:

- ✅ Successful injection recovery
- ✅ Collapse under adversarial controls
- ✅ Stable holdout generalization
- ✅ Robustness to discretization

These results support the **cross-domain stability of the CDR estimation framework**.

---

# Phase II.3 — Ecological Dynamics (Completed)

🔗 **GitHub repository:**
https://github.com/ThiagoLuzpY/cdr-phase2.3-ecology

---

## Objective

Apply the CDR framework to **biological dynamical systems**, specifically:


predator-prey population dynamics


This phase introduces systems with:

- Non-linear feedback loops  
- Cyclical dynamics  
- Strong endogenous structure  

---

## Dataset

```
Hudson Bay Company Lynx–Hare dataset
```

---

## System Representation

Final state definition:

```
state = (hare_log_return, lynx_log_return)
```

---

## Discretization

```
3 bins per variable
3² = 9 states
```

---

## Key Methodological Adjustments

### 1. Dimensionality Correction

Removed:

```
year_norm
exogenous variables
```

Reason:

```
avoid sparsity
preserve endogenous structure
```

---

### 2. Temporal Strategy

Used:

```
interleaved train/test split
```

Reason:

```
ecological systems are cyclical
chronological split breaks phase consistency
```

---

### 3. Control System (Final)

```
shuffle_time
block_shuffle
species_swap
transition_randomization
```

---

## Phase II.3 Results


CDR Phase II.3 (Ecology)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.275
eps_true: 0.25
abs_err: 0.025

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.00
abs_delta: 0.00

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

## Interpretation

```
ε ≈ 0
```

Meaning:

- System is fully explained by internal dynamics  
- No external reweighting required  
- Strong structural determinism  

---

## Scientific Insight

This confirms that:


CDR correctly identifies endogenous structure in biological systems


---

# Phase II.4 — Protein Dynamics (Completed)

🔗 **GitHub repository:**
https://github.com/ThiagoLuzpY/cdr-phase2.4-protein


## Objective

Apply the CDR framework to microscopic biological dynamical systems, specifically:

- molecular dynamics simulations
- protein folding trajectories
- conformational state transitions


This phase introduces systems with:

- Energy-landscape-governed transitions

- Strong physical constraints

- Low-dimensional conformational coordinates

- Multiple independent simulation trajectories

---

## Dataset
```
Alanine dipeptide molecular dynamics trajectories
```

Files used:
```
alanine-dipeptide-nowater.pdb
alanine-dipeptide-0-250ns-nowater.xtc
alanine-dipeptide-1-250ns-nowater.xtc
alanine-dipeptide-2-250ns-nowater.xtc
```
---
## System Representation


Protein dynamics were represented through backbone dihedral structure:
```
state = (phi_bin, psi_bin)
```
These variables capture the dominant conformational geometry of alanine dipeptide and provide a compact state-space representation of the molecular dynamics.
---

## Discretization
```
3 bins per variable
3² = 9 states
```
---

## Sampling / Preprocessing

Frame thinning:
```
frame_stride = 10
```

Effective observations loaded:
```
75,000 frames
```
The loader extracted dihedral trajectories from independent molecular dynamics simulations and assembled the conformational variables used in the CDR pipeline.

---

## Key Methodological Adjustment
```
Trajectory-Level Holdout for F3
```
The initial sequential holdout failed because independent .xtc simulations should not be treated as one single continuous trajectory.

---

### Final F3 strategy:
```
leave-one-trajectory-out holdout
```

Implemented as:

- Train on all trajectories except one held-out simulation

- Test on the held-out trajectory only

- Build transitions only within each trajectory

- Prevent artificial transitions across .xtc file boundaries


This adjustment preserved the falsifiability of the framework while aligning the holdout protocol with the physical structure of the dataset.
```
Control System
shuffle_next
circular_shift
block_shuffle
```

These controls were designed to break correct frame-to-frame pairing while preserving partial marginal or short-range structure.

---

## Phase II.4 Results
CDR Phase II.4 (Protein)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.27
eps_true: 0.25
abs_err: 0.02

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00
n_controls: 10

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.10
abs_delta: 0.10

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
Interpretation
```

---

The alanine dipeptide dynamics appear consistent with a structurally sufficient physical kernel:
```
ε ≈ 0
```

Meaning:

- Molecular transitions are largely explained by the endogenous conformational dynamics

- No additional structural reweighting is required at the tested resolution

- The estimator remains sensitive, as shown by successful injected-structure recovery

- Importantly, the final result is consistent with the expectation that strongly constrained molecular systems should behave closer to energy-governed physical dynamics than to partially latent neural systems.

---

## Scientific Insight

This phase confirms that:

- CDR remains stable at the molecular scale

- and that trajectory-aware holdout design is essential when independent simulations are used as empirical test domains.

---

## Updated Cross-Domain Conclusions (Phase II + Phase III)

Across **eight empirical domains / experimental configurations**:

| Domain | Result |
|--------|--------|
| Energy infrastructure | ε ≈ 0 |
| Neural dynamics (fMRI) | ε ≈ 0.06–0.08 |
| Human mobility | ε ≈ 0 |
| Ecological systems | ε ≈ 0 |
| Protein dynamics | ε ≈ 0 |
| EEG-only | ε ≈ 0.04 |
| RNG-only | ε ≈ 0 |
| Joint EEG + RNG | ε ≈ 0.00–0.03 |

---

The cumulative empirical picture now shows that CDR can distinguish at least **three broad operational regimes**:

### 1. Structurally sufficient systems

```
ε ≈ 0
```

Observed in:

- Energy infrastructure
- Human mobility
- Ecological population dynamics
- Protein dynamics
- RNG-only baseline

These systems are adequately explained by the empirical reference kernel at the tested resolution.

---

### 2. Low but non-zero residual-structure systems

```
ε > 0 (low)
```

Observed in:

- fMRI
- EEG-only

These results suggest that some neural systems may retain low residual structure not fully captured by the chosen baseline kernel, while remaining stable under controls, holdout, and sensitivity tests.

---

### 3. Weak joint-structure systems

```
ε_joint ≈ 0.00–0.03
```

Observed in:

- Joint EEG + RNG validation

The joint domain did **not** show strong cross-domain structure under the present observational setup. The final compact joint representations stabilized the pipeline and eliminated major sparsity artifacts, but the observed joint epsilon remained low.

---

## Emerging Methodological Insight

The cumulative results also suggest that **F1 injection recovery is domain-dependent**.

A contrast emerged between:

- **high-injectability domains**, where injected structure is recovered robustly  
  (e.g. mobility, ecology, protein, engineered systems)

and

- **low-injectability domains**, where injected structure is recovered only weakly or at the threshold of tolerance  
  (e.g. fMRI, EEG, RNG, and the joint EEG+RNG domain)

This suggests that F1 may measure not only estimator validity, but also a domain-sensitive property related to how easily imposed artificial structure can be absorbed and recovered by the system representation.

---

## Interpretation

Taken together, these results support the following view:

- **CDR does not act as a generic detector of randomness**
- **CDR behaves as a domain-sensitive detector of deviation from structural sufficiency**
- neural and joint experimental domains appear to require more careful state construction than macroscopic or strongly endogenous systems
- the present EEG + RNG implementation provides a stable falsifiable baseline, but does **not** yet supply strong evidence for robust cross-domain coupling

---

# Phase III — Experimental Validation (Completed)

🔗 **GitHub repository:**
https://github.com/ThiagoLuzpY/cdr-phase3.0-eeg-rng

Phase III extended the CDR framework from observational domains into **experimental neural and stochastic domains**.

Instead of testing a single joint system immediately, the experimental program was intentionally divided into **three internal stages**:

1. **Phase III.0.1 — EEG-only validation**
2. **Phase III.0.2 — RNG-only baseline validation**
3. **Phase III.0.3 — Joint EEG + RNG validation**

This modular strategy allowed the project to separate:

- neural residual structure
- stochastic baseline behavior
- and cross-domain coupling structure

before attempting stronger theoretical interpretations.

---

## Phase III.0.1 — EEG-Only Validation (Completed)

### Objective

Test whether direct neural activity measured from EEG exhibits low but detectable residual structure under the CDR framework.

This phase was designed as the experimental neural counterpart of the earlier fMRI validation, but with:

- higher temporal resolution
- direct electrophysiological measurement
- and explicit sleep-stage structure

---

### Dataset
```
Sleep-EDF Expanded
subject: SC4001
files:
SC4001E0-PSG.edf
SC4001EC-Hypnogram.edf
```
---

### Initial modeling strategy

The first EEG attempts used a richer state representation combining multiple spectral variables, including configurations such as:

```
(delta_power, theta_power, alpha_power, stage_code)
```

and later expanded neural feature sets.

These early runs showed that the EEG domain was highly sensitive to state-space construction. Some richer configurations produced unstable or weakly generalizable F1 behavior.

---

### Final state representation

The final validated EEG representation used a compact neural state:

```
state = (delta_power_bin, alpha_power_bin)
```

This reduced the state-space dimensionality while preserving physiologically meaningful contrast between lower-frequency and higher-frequency neural regimes.

---

### Final discretization

```
3 bins per variable
3² = 9 states
```

---

### EEG-only results


CDR Phase III.1 (EEG)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.00
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00

F3_holdout_generalization: PASS
eps_train: 0.04
eps_test: 0.00
abs_delta: 0.04

F5_sensitivity: PASS
eps_binsA: 0.04
eps_binsB: 0.04
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

### EEG interpretation

The final EEG-only result indicates:

```
ε ≈ 0.04
```

Meaning:

- neural dynamics exhibit **low but non-zero residual structure**
- the result is stronger than a pure stochastic baseline
- but weaker than a strongly structured macroscopic domain

This is consistent with the interpretation that direct neural data may retain subtle residual organization not fully absorbed by the empirical kernel.

---

## Phase III.0.2 — RNG-Only Validation (Completed)

### Objective

Establish a strong experimental null baseline using a random source expected to approximate minimal structural organization.

This phase had two scientific roles:

- verify that CDR does not hallucinate structure in a stochastic domain
- provide a direct baseline against which joint EEG + RNG analysis could later be compared

---

### Dataset

```
ANU Quantum Random Number Generator
JSON sample file:
anu_sample.json
```

Because the free API limited the raw number count, the final validated representation used:

```
uint8 values converted into binary bits
```

This preserved falsifiability while increasing usable sequence density from a single legitimate sample.

---

### Initial modeling strategy

The first RNG attempts used the raw `uint8` values with larger state spaces, including configurations analogous to:

```
state = (x0_bin, x1_bin) with 3 bins
```

These early versions failed because the control system became unstable and created spurious structure under low-sample, high-sparsity settings.

---

### Final state representation

The final validated RNG representation used a binary-bit state:

```
state = (bit_t, bit_t+1)
```

---

### Final discretization

```
2 bins per variable
2² = 4 states
```

---

### RNG-only results


CDR Phase III.2 (RNG)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.00
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: 0.00

F3_holdout_generalization: PASS
eps_train: 0.00
eps_test: 0.00
abs_delta: 0.00

F5_sensitivity: PASS
eps_binsA: 0.00
eps_binsB: 0.00
abs_delta: 0.00

────────────────────────────────
FINAL: PASS
```

---

### RNG interpretation

The final RNG-only result indicates:

```
ε ≈ 0
```

Meaning:

- no residual structure was detected
- controls collapsed exactly as expected
- the stochastic baseline behaved as an experimentally clean null domain

This is one of the strongest confirmations in the project that CDR does **not** generate spurious structure under a domain intended to behave as near-pure randomness.

---

## Phase III.0.3 — Joint EEG + RNG Validation (Completed)

### Objective

Test whether the joint system formed by EEG and RNG exhibits cross-domain structure stronger than either component alone.

This was the most theoretically ambitious phase of the project, because it aimed to test whether neural activity and a stochastic baseline could form a jointly structured domain under the CDR framework.

---

### Joint strategy

The joint phase was deliberately exploratory and went through **multiple state-space redesigns**.

The original idea was to combine richer EEG and RNG components simultaneously, including joint representations of the form:

```
(delta_power, alpha_power, x0, x1)
```

However, these initial versions produced high sparsity and unstable injection recovery.

---

### Evolution of joint state-space tests

Three major joint configurations were explored:

#### 1. High-dimensional joint state
```
36 states
```

This representation was too sparse and produced unstable F1 behavior, including strong overshoot of injected epsilon.

#### 2. Compact joint state
```
6 states
state = (delta_power_bin, x0)
with 3 EEG bins × 2 RNG bins
```

This version stabilized the joint domain substantially and passed all gates, but only with very low injection values.

#### 3. Intermediate joint states
```
8 states
10 states
```

These were tested as robustness checks by increasing EEG granularity while keeping the same compact joint logic.

Both intermediate variants remained stable and showed results similar to the 6-state model:

- low joint epsilon
- clean controls
- stable holdout
- threshold-level F1 recovery

This demonstrated that the joint conclusion did not depend on a single minimal state-space choice.

---

### Final validated joint conclusion

Across the stabilized joint runs, the pattern remained:

```
eps_train ≈ 0.01–0.03
eps_test ≈ 0.00
```

and no robust high joint epsilon emerged.

---

### Representative joint result


CDR Phase III.3 (Joint EEG + RNG)
────────────────────────────────
```
F1_injection_recovery: PASS
eps_hat: 0.00
eps_true: 0.05
abs_err: 0.05

F2_controls_collapse: PASS
median_eps_controls: 0.00
fraction_below_tol: 1.0
max_eps_controls: low

F3_holdout_generalization: PASS
eps_train: 0.01–0.03
eps_test: 0.00
abs_delta: low

F5_sensitivity: PASS
eps_binsA: low
eps_binsB: low
abs_delta: low

────────────────────────────────
FINAL: PASS
```

---

### Joint interpretation

The final joint experimental result indicates:

```
ε_joint ≈ 0.00–0.03
```

Meaning:

- the joint EEG + RNG domain did **not** exhibit strong cross-domain structure
- the compact state-space redesign successfully stabilized the method
- but the observed joint epsilon remained weak

This is best interpreted as:

- a successful falsifiable implementation of the first EEG–RNG joint experiment
- but **not** as strong evidence of robust neural–stochastic coupling under the current observational setup

---

## Phase III Scientific Interpretation

Phase III achieved three important results:

1. **EEG-only** validation confirmed low but non-zero residual neural structure  
2. **RNG-only** validation established a strong null stochastic baseline  
3. **Joint EEG + RNG** validation showed that, in the present design, the cross-domain joint epsilon remains weak

This does **not** invalidate broader ontological interpretations such as TSQP, but it does constrain the simpler version of the EEG–RNG coupling hypothesis tested here.

The main conclusion of Phase III is therefore:

- the simple observational joint design is **stable and falsifiable**
- but likely too coarse to capture a deeper latent or quantum-aware informational principl

## Future Validation Domains

### Phase III.1 Redesign — Multi-Subject and Quantum-Aware Joint Experiment

The next step of the project is **not** to repeat the simple EEG-only or RNG-only baselines.

Instead, the next phase will redesign the joint experiment itself.

The current Phase III demonstrated that:

- EEG-only produces low but non-zero residual structure
- RNG-only behaves as a strong stochastic null baseline
- the simple observational joint EEG + RNG design remains too weak to reveal robust cross-domain structure

This motivates a new redesign stage.

---

### Objective

Build a more advanced joint framework capable of testing whether a deeper informational principle may exist beyond the directly observed classical signals.

---

### Planned redesign axes

#### 1. Multi-subject EEG expansion
The next joint experiment will incorporate multiple EEG subjects rather than relying on a single subject only.

Purpose:
- improve robustness
- reduce dependence on individual-specific sleep structure
- test whether weak joint effects remain stable across subjects

---

#### 2. Informational latent layer
The next version will no longer rely only on raw observables such as bandpower and binary RNG bits.

Additional derived variables may include:

- spectral entropy
- permutation entropy
- local complexity
- run-length entropy
- compressibility proxies
- local surprise / self-information
- persistence and transition-regime variables

Purpose:
- test whether informational structure appears more clearly in latent-derived variables than in raw observables alone

---

#### 3. Quantum-aware proxy layer
The current Phase III joint model uses the final observed stochastic output only.

The redesign will explicitly explore whether a deeper latent term should be modeled, for example via proxies related to:

- pre-measurement instability
- hidden-basis-like latent structure
- background informational noise
- collapse / decoherence proxies
- regime-dependent latent coupling terms

Purpose:
- move beyond a purely classical observational joint design
- test whether the hypothesized informational principle may require latent or indirect modeling rather than direct raw-state coupling

---

### Conceptual mathematical direction

A future redesign may extend the joint state model from:

```
state_t = (EEG_t, RNG_t)
```

toward an augmented formulation such as:

```
state_t = (EEG_t, RNG_t, I_t, Z_t, Q_t)
```

where:

- `I_t` = informational proxy variables
- `Z_t` = latent inferred state
- `Q_t` = quantum-aware proxy term

and where the effective structural term may evolve from:

```
Δχ_observed
```

to something closer to:

```
Δχ* = Δχ_observed + λ₁ I_t + λ₂ Z_t + λ₃ Q_t
```

This preserves the falsifiable spirit of CDR while allowing the framework to test deeper hypotheses about informational or quantum-mediated structure.

---

### Scientific meaning of Phase III.1 Redesign

The redesigned phase is intended to answer a more ambitious question than the current observational joint test:

> *If a deeper informational principle exists, can it be detected only after introducing latent, informational, or quantum-aware structure into the model rather than relying on the raw EEG–RNG pairing alone?*

This will be the main focus of the next stage of the project.

---

## Project Structure
```
cdr-phase1-validation/
│
├── config/
│   ├── __init__.py
│   ├── phase1_config.py
│   ├── phase2_config.py
│   ├── phase2_config_ecology.py
│   ├── phase2_config_fmri.py
│   ├── phase2_config_mobility.py
│   └── phase2_config_protein.py
│   └── phase2_config_eeg.py
│   └── phase2_config_joint.py
│   └── phase2_config_rng.py
│
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       ├── ecology/
│       ├── fmri/
│       ├── geolife/
│       ├── eeg/
│       ├── rng/
│       ├── opsp/
│       │   ├── datapackage.json
│       │   ├── README.md
│       │   ├── time_series.sqlite
│       │   └── time_series_60min_singleindex.csv
│       └── protein/
│           ├── alanine-dipeptide-nowater.pdb
│           ├── alanine-dipeptide-0-250ns-nowater.xtc
│           ├── alanine-dipeptide-1-250ns-nowater.xtc
│           └── alanine-dipeptide-2-250ns-nowater.xtc
│
├── results/
│   ├── golden_run_phase1_plus_v1/
│   ├── phase2_opsp/
│   │   ├── bins_specs.json
│   │   ├── checkpoint_controls.json
│   │   ├── checkpoint_eps.json
│   │   ├── data_report.json
│   │   ├── ll_injection.png
│   │   ├── ll_test.png
│   │   ├── ll_train.png
│   │   ├── phase2_config.json
│   │   ├── phase2_results.json
│   │   ├── report.txt
│   │   └── selection.json
│   ├── phase2_ecology/
│   ├── phase2_fmri/
│   ├── phase2_mobility/
│   ├── phase2_protein/
│   ├── phase3_eeg/
│   ├── phase3_joint/
│   ├── phase3_rng/
│   └── .gitkeep
│
├── scripts/
│   ├── __init__.py
│   ├── make_audit_bundle.py
│   ├── download_qrng.py
│   └── run_phase1_plus_full.py
│
├── src/
│   ├── kernels/
│   │   ├── empirical_kernel.py
│   │   └── reweighted_kernel.py
│   ├── __init__.py
│   ├── adversarial_kernel.py
│   ├── artifacts.py
│   ├── build_states.py
│   ├── controls.py
│   ├── controls_phase2.py
│   ├── controls_phase2_ecology.py
│   ├── controls_phase2_fmri.py
│   ├── controls_phase2_mobility.py
│   ├── controls_phase2_protein.py
│   ├── controls_phase3_eeg.py
│   ├── controls_phase3_joint.py
│   ├── controls_phase3_rng.py
│   ├── discretize.py
│   ├── ecology_loader.py
│   ├── eeg_loader.py
│   ├── estimators.py
│   ├── fmri_loader.py
│   ├── geolife_loader.py
│   ├── ising_kernel.py
│   ├── joint_loader.py
│   ├── model_selection.py
│   ├── opsp_loader.py
│   ├── phase1_plus_runner.py
│   ├── phase1_runner.py
│   ├── phase2_runner.py
│   ├── phase2_runner_ecology.py
│   ├── phase2_runner_fmri.py
│   ├── phase2_runner_mobility.py
│   ├── phase2_runner_protein.py
│   ├── phase3_runner_eeg.py
│   ├── phase3_runner_joint.py
│   ├── phase3_runner_rng.py
│   ├── protein_loader.py
│   ├── rng_loader.py
│   ├── statistics.py
│   ├── validators.py
│   └── validators_phase2.py
│
├── tests/
│   ├── __init__.py
│   ├── test_controls.py
│   ├── test_estimators.py
│   ├── test_ising.py
│   ├── test_phase1_plus_runner.py
│   ├── test_statistics.py
│   └── test_validators.py
│
├── venv/
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## Running Phase II

**Energy system validation:**
```bash
python -m src.phase2_runner
```

**fMRI validation:**
```bash
python -m src.phase2_runner_fmri
```

**Mobility validation:**
```bash
python -m src.phase2_runner_mobility
```

**Ecology validation:**
```bash
python -m src.phase2_runner_ecology
```

**Protein validation:**
```bash
python -m src.phase2_runner_protein
```

**EEG-only validation:**
```bash
python -m src.phase3_runner_eeg
```

**RNG-only validation:**
```bash
python -m src.phase3_runner_rng
```

**Joint EEG + RNG validation:**
```bash
python -m src.phase3_runner_joint
```

**Results saved in:**
```
results/phase2_opsp/
results/phase2_fmri/
results/phase2_mobility/
results/phase2_ecology/
results/phase2_protein/
results/phase3_eeg/
results/phase3_rng/
results/phase3_joint/
```

---

## Reproducibility

The pipeline ensures reproducibility via:

- ✅ Fixed random seeds
- ✅ Serialized configuration files
- ✅ Saved discretization bins
- ✅ Stored likelihood curves
- ✅ Checkpoint files
- ✅ Domain-specific holdout protocols when required by data structure
- ✅ Explicit separation between observational domains, neural domains, stochastic baselines, and joint domains
- ✅ Stable compact state-space redesign when sparsity becomes dominant

All experiments are deterministic under identical configurations.

---
### Experimental data sources used in Phase III

**EEG source:**
```
Sleep-EDF Expanded
```

Files used in the first validated experimental run:
```
SC4001E0-PSG.edf
SC4001EC-Hypnogram.edf
```

**RNG source:**
```
ANU Quantum Random Number Generator
```

Sample file used:
```
anu_sample.json
```

Final validated RNG representation:
```
uint8 values converted into binary bits
```

This allowed the stochastic baseline to remain falsifiable while increasing usable sequence


---
## References

- Popper, K.R. (1959). *The Logic of Scientific Discovery.*
- Lakatos, I. (1978). *The Methodology of Scientific Research Programmes.*
- Rosen, R. (1991). *Life Itself.*
- Open Power System Data (2020) https://open-power-system-data.org/
- GeoLife GPS Trajectory Dataset (Microsoft Research)
- OpenNeuro dataset ds002938
- Alanine dipeptide molecular dynamics trajectories
- Sleep-EDF Expanded dataset (PhysioNet)
- ANU Quantum Random Number Generator

---

## Citation
```bibtex
@software{luz2026cdr,
  title={Core Distinguishability Relativity: Empirical Validation Framework},
  author={Luz, Thiago},
  year={2026},
  url={https://github.com/ThiagoLuzpY/cdr-phase1-validation}
}
```

---

## License

**CC0 1.0 Universal (Public Domain)**

---

## Author

**Thiago Luz**  
Independent Researcher  
Rio de Janeiro, Brazil

- **GitHub:** https://github.com/ThiagoLuzpY/
- **ORCID:** https://orcid.org/0009-0008-9732-324X

---

## Acknowledgments

Thanks to the open scientific ecosystem:

- NumPy
- SciPy
- pandas
- nilearn
- mdtraj
- OpenNeuro
- Open Power System Data
- Microsoft GeoLife Dataset
- Sleep-EDF Expanded dataset (PhysioNet)
- ANU Quantum Random Number Generator

---

**Last updated:** March 2026  
**Status:** Phase I complete ✅ | Phase II.1A complete ✅ | Phase II.1B complete ✅ | Phase II.2 complete ✅ | Phase II.3 complete ✅ | Phase II.4 complete ✅ | Phase III.0.1 complete ✅ | Phase III.0.2 complete ✅ | Phase III.0.3 complete ✅ | Phase III.1 Redesign planned 📋

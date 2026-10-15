# GNN-practical-assignments
# Technical Report: GNN Practical Assignments

Values marked **⟦TODO⟧** must be filled from your own notebook output. Values marked "≈" were read off plots; replace them with the exact numbers from your tables.

## Summary of results

| # | Task | Headline result |
|---|---|---|
| 1 | GCN from scratch (synthetic) | Manual layer check and permutation-equivariance check pass; GCN vs MLP and edge ablation ⟦TODO⟧ |
| 2 | MLP / GCN / GraphSAGE / GAT on Cora | GCN best: test acc 0.820 ± 0.004, macro F1 0.807; MLP 0.587 |
| 3 | Link prediction + leakage | MLP decoder AUC 0.949 / AP 0.955 vs dot 0.916 / 0.921 vs common neighbours 0.721 / 0.719; leaky run inflates scores |
| 4 | MLOps pipeline | 4 tracked runs, gated promotion, API on unseen nodes (acc 0.810 vs 0.829 seen), drift detected at ≈ 10× (graph) and ≈ 20× (embedding) the normal level |

## Reproducibility

- **Environment:** Python ⟦x.y.z⟧, torch ⟦…⟧, torch_geometric ⟦…⟧, scikit-learn ⟦…⟧, mlflow-skinny ⟦…⟧, fastapi ⟦…⟧ (`pip freeze`).
- **Seeds:** data and split seed 0 (A1, A3); A2 final runs seeds 0/1/2, tuning seed 0; A3 seeds 0/1/2; A4 tracked runs seed 0, CI gate seed 1.
- **Commands:** see `README.md`. Every notebook runs top to bottom and has saved outputs; each assignment has automated tests.

---

## 1. Assignment 1: GCN from scratch

### Dataset design
- 200 nodes, 3 numerical features, 2 balanced classes, labels assigned at random.
- **Features:** class mean + N(0,1) noise. Means differ only slightly (shift 0.6), so one node's features are weakly informative; the third feature is pure noise. Labels are never an input feature.
- **Edges (stochastic block model):** same-class pairs connect with p = 0.08, different-class pairs with p = 0.01, so neighbours carry label information beyond a node's own features (edge homophily ≈ ⟦TODO⟧).
- **Split:** 30 % train / 20 % validation / 50 % test, random (seed 0), asserted disjoint.

### Implementation
Self-loops, degree normalisation 1/√(d_i d_j), message passing (`index_add_` over edges) and the two-layer GCN use PyTorch tensor operations only. The training loop (cross-entropy on train labels, backpropagation, Adam update) keeps the best-validation checkpoint; test accuracy is computed once at the end.

### Correctness checks
1. **Manual layer check:** one node's layer-1 output equals a plain Python loop over its neighbours with identical weights (match: ⟦True⟧).
2. **Permutation equivariance:** relabelling nodes and shuffling edge order gives identically permuted outputs (max abs difference ⟦…⟧ < 1e-5).

### Results (5 seeds, same features and split)
| Model | Test accuracy (mean ± std) |
|---|---|
| MLP (no edges) | ⟦TODO⟧ |
| GCN, real edges | ⟦TODO⟧ |
| GCN, random edges | ⟦TODO⟧ |
| GCN, self-loops only | ⟦TODO⟧ |

### Does the model use the graph?
The GCN should beat the MLP and fall towards its level when edges are randomised or removed; the hidden-layer plot should show message passing pulling same-class nodes together. ⟦Write your conclusion from your numbers.⟧

---

## 2. Assignment 2: Architecture comparison on Cora

### Dataset
Cora: 2708 papers (nodes), 1433 binary bag-of-words features (row-normalised), 5278 undirected citation edges (average degree ≈ 4), 7 classes, strongly homophilous (≈ 81 % of edges join same-class nodes). Standard Planetoid split: 140 train (20 per class) / 500 validation / 1000 test. All nodes and edges are visible in every forward pass; only the 140 training labels supervise the loss. Validation is used for tuning and checkpoints; test only for final reporting.

### Experiment design
- Own model classes (MLP, GCN, GraphSAGE, GAT) on PyG layers; own training and evaluation code.
- Equal tuning budget: 8 random configurations (learning rate, weight decay, hidden size, dropout), identical for every model, selected on validation accuracy. Best GCN: lr 0.02, wd 1e-3, hidden 32, dropout 0.5 (val 0.812).
- Final runs: 3 seeds, mean ± std (population std).

### Results
| Model | Test acc | Macro F1 | Train time (s) | Parameters |
|---|---|---|---|---|
| MLP | 0.587 ± 0.003 | 0.566 ± 0.002 | 18.0 | 46,119 |
| **GCN** | **0.820 ± 0.004** | **0.807 ± 0.002** | 13.5 | 46,119 |
| GraphSAGE | 0.802 ± 0.005 | 0.794 ± 0.004 | 34.1 | 92,199 |
| GAT | 0.807 ± 0.013 | 0.800 ± 0.013 | 23.9 | 23,109 |

All GNNs beat the MLP by about 20 points. GCN vs GAT (0.013) is about one GAT standard deviation, so not clearly significant with 3 seeds; GCN vs GraphSAGE is small but consistent. Training time mostly reflects early stopping (the MLP ran all 200 epochs, GCN stopped near epoch 80), so per-epoch cost is the fairer comparison. MLP and GCN have equal parameter counts because a GCN layer is a linear layer plus fixed neighbour averaging.

**Why GCN is best here (reasoning, not proven):** Cora is strongly homophilous and labels are scarce, so fixed degree-normalised averaging uses neighbour information without extra capacity to overfit. GraphSAGE adds separate self/neighbour weights (twi
ce the parameters) and GAT adds attention that must be learned from 140 labels, with little to gain because most neighbours are equally informative. GCN was also the most hyperparameter-sensitive (validation 0.62–0.81 across trials), so the small tuning budget matters.

### Depth study (2–8 layers, 3 seeds)
- GCN: 0.820 (2 layers) → about 0.76–0.78 (6–8). GraphSAGE: 0.803 → 0.758. GAT collapsed at 8 layers (≈ 0.56, very high variance), pointing to optimisation difficulty in deep attention stacks without residuals.
- Mean pairwise cosine similarity of last-hidden embeddings **decreased** with depth (GCN 0.645 → ≈ 0.15–0.2), the opposite of classic over-smoothing. Probably a property of the metric: shallow ELU embeddings share a large common component. A centred cosine or MAD would be needed to confirm.

### Edge removal (train and test on the reduced graph; MLP is the control)
| Fraction removed | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| MLP | 0.587 | 0.587 | 0.587 | 0.587 | 0.587 | 0.587 |
| GCN | 0.820 | 0.799 | 0.767 | 0.696 | 0.648 | 0.583 |
| GAT | 0.810 | 0.783 | 0.753 | 0.688 | 0.637 | 0.571 |
| GraphSAGE | 0.802 | 0.766 | 0.715 | 0.610 | 0.590 | 0.591 |

The MLP is flat as a control should be. At 100 % removal every GNN falls to MLP level. GraphSAGE degrades fastest (hypothesis: its separate neighbour channel is noisier on thin neighbourhoods).

### Error analysis (five most confident wrong GCN predictions)
| Node | True → pred | MLP pred | Conf | Degree | Neighbour true labels | Words | Reading |
|---|---|---|---|---|---|---|---|
| 1990 | 2 → 3 | 3 | 0.60 | 1 | {3} | 5 | single neighbour of another class; both models wrong |
| 1774 | 4 → 1 | 1 | 0.57 | 1 | {1} | 25 | single neighbour of another class; both wrong |
| 1764 | 5 → 2 | **5** | 0.48 | 2 | {5, 2} | 21 | MLP right; neighbours were themselves predicted 2 |
| 2323 | 4 → 5 | 5 | 0.45 | 2 | {5, 2} | 5 | heterophilous neighbours, few words |
| 2484 | 3 → 4 | 0 | 0.44 | 3 | {4, 4, 4} | 18 | all neighbours class 4 |

All five have degree 1–3, so one neighbour dominates aggregation; four have neighbourhoods dominated by other classes; confidence is low (0.44–0.60), so a threshold could flag them. The confusion matrix shows class 0 over-predicted (precision ≈ 0.61) and classes 5 and 6 with the lowest recall.

### Recommendation
A 2-layer GCN: best accuracy and macro F1, lowest GNN seed variance, same parameters as the MLP and the cheapest GNN. GraphSAGE and GAT gave no measurable benefit on this small homophilous graph. Use the MLP only if edges are unavailable or unreliable.

**Limitations:** one public split, 3 seeds, 8 tuning trials per model, std with ddof = 0.

---

## 3. Assignment 3: Link prediction and data leakage

### Pipeline
- Unique undirected links split 85 / 5 / 10 % into train / validation / test (stored once with u < v; both directions always in the same split).
- Message-passing graph = training links in both directions, used for training, validation **and** testing.
- Negatives: random pairs with no self-links and none among observed links. Validation and test negatives are one fixed pool cut in two (no overlap). Training negatives are resampled each epoch but avoid all observed links and all evaluation negatives.
- Encoder: 2-layer GCN (no category labels). Decoders: dot product, and an MLP on `[zu*zv, |zu−zv|]`, which is exactly symmetric in the pair.
- Baseline: common neighbours counted on training links only.

### Automated checks
Held-out links and their reverses are absent from the message-passing graph; positive splits are disjoint; negatives have no self-links, observed links, duplicates or overlap; the message-passing graph equals the training links and is symmetric; both decoders score reversed pairs identically. The checker is itself tested: it rejects the leaky graph, a graph with one direction of a test link and one with only a reverse copy.

### Hyperparameter finding (validation only)
| Weight decay | Dot val AUC | MLP val AUC |
|---|---|---|
| 5e-4 | 0.775 | 0.799 |
| 0 | 0.915 | 0.951 |

The 5e-4 inherited from node classification caused underfitting; likely L2 shrinks embedding norms and therefore score magnitudes. Selection used validation only.

### Results (valid setting)
| Method | Val AUC | Test AUC | Test AP |
|---|---|---|---|
| Common neighbours | 0.712 | 0.721 | 0.719 |
| Dot decoder | 0.915 | 0.916 | 0.921 |
| **MLP decoder** | **0.952** | **0.949** | **0.955** |

Validation and test agree within 0.004, so selection did not overfit validation. The MLP decoder wins likely because it learns its own scoring rule (it also sees |zu − zv|), while the dot product only measures alignment. Common neighbours is limited because Cora is sparse and many held-out pairs have no shared training neighbour, tying with most negatives at score 0. Mean ± std over 3 seeds: ⟦from `compare()`⟧.

### Leaky experiment
Same fixed settings, but held-out positives are added to the message-passing graph (never used for model selection). Test positives collapse onto probability ≈ 1 while negatives barely change: the encoder detects the edge that is present instead of predicting a missing one. Leaky vs valid test AUC: ⟦leaky ___ vs valid 0.949⟧. The leaky evaluation is invalid because the link to be predicted is, by definition, absent from the graph at prediction time.

### Failure cases
The five lowest-scored true links (logits −10 to −16) all had 0 common training neighbours; three involved an endpoint with training degree 0 (cold start: embedding from its own words only). The highest-scored false alarms had moderate degree (2–8) and at most one common neighbour, suggesting the model scores feature and neighbourhood similarity. Errors are confident in both directions, so scores rank well but are not calibrated probabilities. The decoder never sees common-neighbour counts; shared neighbours help only indirectly through the encoder.

### Candidate links
The ten highest-scored unobserved pairs (excluding every link in any split) are in the notebook. An unobserved pair is not a confirmed negative: the citation may be missing from the dataset, unknown to the authors, or appear later. The list is a set of hypotheses needing external validation (reading the papers, later citation data, expert review). Shared category is weak supporting evidence only (labels were not used in training).

**Limitations:** random negatives are mostly easy, so AUC/AP are optimistic; the failure tables show worst cases, not frequency.

---

## 4. Assignment 4: MLOps pipeline

The pipeline extends Assignment 2's GCN (Cora node classification). Data flow: train → track → package → register and gate → serve → monitor → rollback; CI repeats the gate on every push.

### 4.1 Untracked baseline and its limitations
`baseline/train_bare.py` trains a GCN and prints one accuracy. It has no seed (results change every run), no record of parameters, metrics, code or data version, no validation-based checkpointing, overwrites `model.pt` each run (no versions), saves no preprocessing or label mapping, cannot verify a reload, has no comparison with a previous model, no API, no monitoring and no rollback.

### 4.2 Experiment tracking (MLflow, SQLite store)
Four runs, each logging parameters, seed, git commit tag, dataset hash, per-epoch loss and validation accuracy, and the model package as an artifact.

| Run | Setting | Best val acc | Behaviour in the curves |
|---|---|---|---|
| weak-40-labels | 40 training labels | ≈ 0.59 | learns slowly, plateaus ≈ 0.57 |
| gcn-h16 | hidden 16, 140 labels | ≈ 0.785 | reaches ≈ 0.77 after about 40 epochs |
| **gcn-h64** | hidden 64, 140 labels | **≈ 0.805** | fastest, ≈ 0.80 by epoch 50 |
| degraded-7-labels | 7 training labels | ≈ 0.41 | training loss → 0 (overfits 7 labels); validation stalls at ≈ 0.32 (majority-class level) |

The weak and degraded runs exist on purpose: they give the gate a clearly worse first model and a clearly regressed candidate. Exact values: ⟦paste the MLflow table with seed, git commit and dataset id⟧.

### 4.3 Model packaging
A package contains `weights.pt` and `meta.json` (architecture, preprocessing description, feature schema of 1433 non-negative floats, label mapping, dataset id, torch version, model version). The row-normalisation is applied **inside** `Predictor`, so callers send raw features. Reloading the package reproduces the original logits exactly (`torch.equal`), and two independent reloads agree. Label names are placeholders `class_0`–`class_6` ⟦replace with real topic names if available⟧.

### 4.4 Registry and validation gate
Metric: validation accuracy of candidate **and** approved model, recomputed on the same validation set.

| Situation | Decision |
|---|---|
| Candidate metric missing / NaN | block |
| Below the floor (0.40) | block |
| No approved model (first model) | promote |
| Approved model's metric missing | block (cannot compare) |
| Gain ≥ 0.005 | promote |
| Gain in [0, 0.005) (tie / noise) | block, keep approved |
| Gain < 0 | block (regression) |

Demonstration (confirm exact reasons from your notebook output):

| Version | Run | Val acc | Decision |
|---|---|---|---|
| v1 | weak-40-labels | ≈ 0.59 | promoted (first model, above floor) |
| v2 | gcn-h16 | ≈ 0.785 | promoted (large gain) |
| v3 | gcn-h64 | ≈ 0.805 | promoted (gain ≈ +0.02 ≥ 0.005); the API served `v3` |
| v4 | degraded-7-labels | ≈ 0.41 | **blocked** (regression, and at the floor) |

A failed evaluation of any model counts as a missing metric and blocks promotion. A failed gate never changes the approved pointer.

### 4.5 CI and rollback
`.github/workflows/ci.yml` runs the tests and `scripts/gate.py` on every push. The script trains a small candidate, compares it with the approved model on the same validation data in a temporary copy of the registry, and exits with code 1 when blocked, which fails the job. CI uses `--min-delta -0.02` ("no significant regression") because an identical re-trained model would tie and otherwise fail every push. Demonstration: `python scripts/gate.py` passes; `python scripts/gate.py --n-labels 7` is blocked (exit code 1). GitHub run links: ⟦green run⟧ ⟦red run⟧.
Rollback: `Registry.rollback()` restores the previous approved version from the history, and `POST /reload` makes the running service switch without a restart. Observed: ⟦before: v3 → after rollback: v2 → restored: v3⟧.

### 4.6 API, Docker and unseen nodes
**API:** FastAPI with `/predict` (features, edges, target ids → label, confidence, `isolated` flag, model version), `/health` and `/reload`. Target ids are row numbers inside the request. Docker image serves the approved model from `registry/`: ⟦docker build and `/health` screenshot⟧.

**Unseen nodes:** 200 test nodes and all their incident edges were removed from the training graph; at serving time the full graph is supplied.

| | Accuracy | Mean confidence | Confidence when right | Confidence when wrong |
|---|---|---|---|---|
| Seen test nodes | 0.829 | 0.447 | 0.474 | 0.314 |
| Unseen nodes | 0.810 | 0.416 | 0.446 | 0.289 |

The drop is within sampling noise (200 nodes ≈ ±3 points), so the model generalises to nodes introduced after training. Confidence is far below accuracy, so probabilities are **underconfident** (likely small logits from row-normalised inputs, weight decay and dropout), but they still separate correct from wrong predictions (0.47 vs 0.31), so they work for ranking and thresholding; temperature scaling would calibrate them.

**Graph-context requirement.** Each request must carry the targets' features and neighbourhood. A 2-layer GCN reproduces full-graph predictions exactly only with the **3-hop** neighbourhood (two hops of messages plus the degrees of the outermost nodes). Agreement with the full-graph prediction on 40 unseen nodes: 1 hop ≈ 0.95, 2 hops ≈ 0.97, 3 hops = 1.00, so too little context changes normalisation and aggregation.
**Limitations:** an isolated node is predicted from its own features only (flagged); features must match the 1433-dimensional non-negative schema; classes never seen in training cannot be predicted; the model reflects the training graph's degree and feature distribution (see drift).

### 4.7 Failure handling (expected behaviour, tested)
| Case | Behaviour |
|---|---|
| Isolated node | 200, predicted from own features, `isolated: true` |
| Duplicate target ids | 422 |
| Invalid edge endpoint (out of range, negative, malformed) | 422 |
| Wrong feature dimension / negative or non-finite values | 422 |
| Unknown or empty target ids | 422 |
| Missing or corrupt model | 503 on `/predict` and `/health` |

### 4.8 Prediction consistency
For node 1708 the API returned class 1 with confidence 0.1855 and the offline model on the full graph 0.186 (same class, difference < 1e-4). The notebook asserts agreement for 10 nodes (5 seen, 5 unseen) between the API, the offline model on the same payload, and the offline model on the full graph. (Node 1708 is a hard case: its probabilities were nearly flat, 0.09–0.19, and the true class 3 received 0.156.)

### 4.9 Drift detection
Fixed model = approved version. **Graph statistic:** relative change of mean degree. **Embedding statistic:** 1 − mean cosine similarity between old and new hidden embedding of each node. **Thresholds:** max(mean + 3σ, 1.5 × mean) over ten *normal* snapshots (5 % random edges missing), which sits above normal fluctuation and far below real change.

| Statistic | Threshold | Normal snapshot (5 % edges removed) | Drifted snapshot (50 % edges, 30 % words removed) |
|---|---|---|---|
| Degree change | ≈ 0.077 | ≈ 0.048, **no alert** | ≈ 0.51, **alert** (≈ 7× threshold) |
| Embedding shift | ≈ 0.0009 | ≈ 0.0005, **no alert** | ≈ 0.018, **alert** (≈ 20× threshold) |

The absolute embedding shift is small because the model is robust and the hidden layer averages over neighbours; what matters is the ratio to normal variation. Both statistics flag the drifted snapshot and neither flags the normal one. **Limitations:** paired comparison needs the same node ids in both snapshots; thresholds come from one kind of normal variation (random edge loss).

**Why raw embeddings of different model versions are not comparable.** Hidden units have no fixed meaning: permuting or rotating them yields an equivalent model, and different seeds or versions land in different coordinate systems. Demonstration: permuting hidden units leaves predictions unchanged while the paired cosine of the raw embeddings drops (⟦value⟧) and linear CKA stays ≈ 1 (⟦value⟧). Compare outputs, use a rotation-invariant similarity (CKA), or monitor with one fixed model as done here.

### 4.10 Tests
`pytest assignment4/tests` covers package reload, schema/labels/preprocessing, every gate rule, registry promote/block/rollback and persistence, unseen-node graph construction, 3-hop exactness, API success and all failure cases, offline-vs-API agreement, reload after rollback, drift, CKA and MLflow logging. Result: ⟦N passed⟧.

---

## 5. One significant implementation problem

**Problem.** In Assignment 3 the first link predictor scored only about 0.78 validation AUC, barely above the common-neighbour baseline (0.71), although the encoder and decoders were correct and all leakage checks passed.

**Investigation.** The checks ruled out leakage and symmetry bugs, so I suspected training. I had reused weight decay 5e-4 from node classification. I ran a small grid over weight decay and decoder, comparing **validation AUC only**, everything else fixed.

**Resolution.** Weight decay 0 raised validation AUC from 0.775 to 0.915 (dot) and from 0.799 to 0.951 (MLP). Likely L2 shrinks embedding norms and thus score range, causing underfitting; here there are thousands of positive pairs and fresh negatives each epoch, so overfitting is less of a concern than with 140 node labels. I made `wd` an explicit argument, fixed it for the valid and leaky experiments alike, and chose it on validation only.

*(A second problem from Assignment 4: after moving files into `src/`, the editor rewrote imports to `assignment4.src.model`, which broke the notebook with `ModuleNotFoundError`. It was found from the traceback, fixed by restoring plain imports with a search-and-replace, and prevented by disabling import updates on file move. Use whichever problem you prefer.)*

---

## 6. Conclusions
- **A1:** the from-scratch GCN passes both correctness checks; graph use is shown by the MLP comparison and edge ablation ⟦confirm with your numbers⟧.
- **A2:** a 2-layer GCN is the best choice on Cora (0.820 test accuracy); extra capacity in GraphSAGE or GAT does not pay off with 140 labels.
- **A3:** the MLP decoder (AUC 0.949) beats the dot decoder (0.916) and common neighbours (0.721); leakage makes held-out links look trivially predictable and must be prevented by construction and by automated checks.
- **A4:** tracking, packaging, a rule-based promotion gate, rollback, drift monitoring and consistency tests make the model reproducible and safe to promote; the model handles unseen nodes (0.810 vs 0.829 accuracy) given a 3-hop context; known limits are underconfident probabilities and the required graph context.
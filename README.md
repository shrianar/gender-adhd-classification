# gender-adhd-classification
## Gender & ADHD Prediction Using Brain Data

Developed a multi-output **SVM ML pipeline** to predict **biological sex** and **ADHD diagnosis** in **1,213 adolescents** using:
- Resting-state **fMRI connectomes**
- **demographic & psychometric metadata**

Engineered age and categorical features, applying SVM classifiers to predict ADHD and sex.  
Sex-specific models predicting ADHD revealed distinct neural signatures.

---

### Workflow
1. **Data Acquisition** – Healthy Brain Network, Child Mind Institute, Reproducible Brain Charts
2. **Preprocessing** – Missing value imputation, one-hot encoding, standardization
3. **Feature Engineering** – PCA-reduced fMRI features + behavioral/demographic variables
4. **Feature Selection** – RandomForest-based top feature extraction (sex-specific)
6. **Modeling** – Multi-output & single-output SVM classifiers
7. **Evaluation** – Precision, recall, F1-score comparisons across datasets

---


### Code Structure

**`notebooks/`**
- `Master_Notebook_Final.ipynb` – End-to-end pipeline: EDA → preprocessing → feature engineering → modeling → evaluation.
- Step-by-step explanations inline with code outputs.

---

### How It Works (Code Flow)

### Exploratory Data Analysis (EDA)

The **EDA** process began with loading and merging four main datasets:

- **Functional connectomes** (`TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv`)
- **Quantitative metadata** (`TRAIN_QUANTITATIVE_METADATA_new.xlsx`)
- **Categorical metadata** (`TRAIN_CATEGORICAL_METADATA_new.xlsx`)
- **Target variables** (`TRAINING_SOLUTIONS.xlsx`)

A helper function merged these on `participant_id`, and the same steps were applied to test data for consistency.

### 1. Dataset Overview
- **Participants:** 910 in training data after merging
- **Features:** 19,900 connectome features + behavioral, demographic, and clinical variables
- **Targets:**  
  - `ADHD_Outcome` (binary)
  - `Sex_F` (binary: 0 = Male, 1 = Female)

We excluded the connectome block for EDA visualizations due to its high dimensionality.

### 2. Missing Data Analysis
- **Highest missingness**:
  - `MRI_Track_Age_at_Scan`: 30.3%
  - `Barratt_Barratt_P2_Occ`: 17.3%
  - `Barratt_Barratt_P2_Edu`: 15.7%
- Heatmap visualizations revealed that most features had <5% missing data.

**Handling Missing Values:**
- Numeric features: imputed with **mean** or **median**
- Categorical features: imputed with **mode**
- Columns with >50% missingness would be dropped (none exceeded this threshold here)

### 3. Demographic Encoding
Key categorical variables were mapped to human-readable labels:
- **Ethnicity**: 0 = Not Hispanic/Latino, 1 = Hispanic/Latino, etc.
- **Race**: 0 = White, 1 = Black, 2 = Hispanic, ...
- **Scan Location**: 1 = Staten Island, 2 = RUBIC, ...
- **Parental Education**: ordinal values from less than 7th grade to graduate degree

### 4. Distribution Plots
- **Sex distribution**: Males outnumber females
- **ADHD diagnosis**: ~70% positive cases, creating a class imbalance
- **Age distribution**: Median ~11 years; slight younger skew for ADHD-positive group
- **ADHD by race**: Diagnosis rates vary; highest counts in White and Black participants

### 5. Group Comparisons
- **ADHD by Sex:**
  - Males: ~73% ADHD-positive
  - Females: ~63% ADHD-positive
- **Age by ADHD Status:** Minor differences in mean age at scan between ADHD-positive and negative groups
<img width="460" height="679" alt="image" src="https://github.com/user-attachments/assets/1b678bb6-c990-4a74-8c9a-d555e17e71af" />

### 6. Correlation Analysis
- Numeric-only heatmap revealed moderate correlations among certain behavioral scales
- No extreme redundancy warranting feature removal at this stage
<img width="846" height="580" alt="Screenshot 2025-08-21 at 4 15 33 PM" src="https://github.com/user-attachments/assets/a00bcb76-2546-470d-bb0e-7fcf10ea1eef" />

### 7. Connectome Data Note
The fMRI connectome portion consists of **19,900 preprocessed correlation features per participant** (upper triangle of the functional connectivity matrix). These were excluded from visual EDA and will later be **dimensionally reduced via PCA** before modeling.

---

**Key Insights from EDA:**
- There is **label imbalance** for ADHD and sex
- Demographic patterns (race, sex) suggest potential confounding
- High-dimensional connectome data will require **dimensionality reduction**
- Multiple behavioral features show meaningful variance across ADHD outcomes
<img width="481" height="333" alt="Screenshot 2025-08-21 at 4 16 43 PM" src="https://github.com/user-attachments/assets/ded6725a-3f2c-4015-8071-b758b6ce4b6e" />
<img width="476" height="343" alt="Screenshot 2025-08-21 at 4 17 00 PM" src="https://github.com/user-attachments/assets/b69fde8c-da5e-4935-bce3-d4f6be6aef50" />

---

### Key Results
- **ADHD prediction:** **85%** F1-score  
- **Sex prediction:** **92%** F1-score  
These results underscore the value of combining **brain connectivity** with **behavioral metadata** for **fair, interpretable predictions**.

---
### Feature Selection  

- **Data preparation**:  
  - Retained only numeric columns (dropped IDs and raw categorical strings).  
  - Applied **median imputation** for remaining missing values.  

- **Sex-specific modeling**:  
  - Split dataset into **female** (`Sex_F = 1`) and **male** (`Sex_F = 0`) subgroups.  
  - Defined features (`X`) and labels (`y = ADHD_Outcome`) separately for each group.  

- **Random Forest classifiers**:  
  - Trained two independent Random Forest models (one per sex subgroup).  
  - Extracted **feature importances** from each model to identify top ADHD predictors.  

- **Key insights**:  
  - Different predictors emerged as most important for males vs. females.  
  - Behavioral and psychological scales (e.g., **SDQ, APQ, EHQ**) consistently ranked highly across both groups.  
  
<img width="906" height="374" alt="Screenshot 2025-08-21 at 4 18 22 PM" src="https://github.com/user-attachments/assets/34c06723-779a-489f-818d-0b29bd5bca29" />
---

### Feature Engineering  

- **New Features**:  
  - Created **age group buckets** (Child, Pre-Teen, Teen, Young Adult) based on participant scan age.  
  - Added a **Sex × Age interaction term** to capture combined demographic effects.  

- **Categorical Encoding**:  
  - One-hot encoded ethnicity, race, and scan location variables.  
  - Ensured categorical columns were properly typed for encoding.  

- **Dimensionality Reduction**:  
  - Integrated **PCA-reduced fMRI connectome features** with metadata.  
  - Merged by participant IDs to create a combined dataset with ~20,000 features.  

- **Imputation**:  
  - Applied **mean imputation** across all numeric features using `SimpleImputer`.  

- **Model Training & Evaluation**:  
  - Trained a **Random Forest classifier** to predict ADHD outcomes.  
  - Achieved ~**66% accuracy** with high recall for ADHD-positive cases.  
  - Exported **Top 20 most important features** (metadata + PCA components) for further interpretation.  

This stage combined demographic, behavioral, and reduced connectome features into a single dataset, engineered meaningful new variables, and identified the most predictive features for ADHD classification.
<img width="858" height="446" alt="Screenshot 2025-08-21 at 4 18 50 PM" src="https://github.com/user-attachments/assets/8c0d6916-213c-4546-8be5-a99858720cc0" />

---
### Separate Feature Selection by Sex & for Sex Prediction  

- **ADHD Prediction by Sex**:  
  - Built separate Random Forest models for **male** and **female** participants.  
  - Extracted **Top 20 features** per sex group.  
  - Consistently important predictors included behavioral scales (e.g., **SDQ Hyperactivity, SDQ Externalizing**) as well as **fMRI-derived PCA features**.  

- **Sex Prediction**:  
  - Trained a Random Forest classifier on the full dataset to identify the strongest predictors of biological sex.  
  - Top features included **Sex × Age interaction** and multiple PCA-reduced connectome dimensions.  

- **Final Shared Feature Set**:  
  - Merged the top features from male ADHD, female ADHD, and sex prediction models.  
  - Resulted in **57 unique features**, which were exported for downstream modeling.  

- **Test Data Preparation**:  
  - Aligned raw test files (categorical, quantitative, and PCA connectome data) with the training schema.  
  - Applied **mean imputation** and **standard scaling** (fit on training set).  
  - Ensured no target leakage by removing ADHD/sex labels before saving.  
  - Produced a **clean engineered test dataset** ready for evaluation.  
 
<img width="272" height="595" alt="Screenshot 2025-08-21 at 4 20 17 PM" src="https://github.com/user-attachments/assets/9ceb4a56-31a7-4e5a-a208-e7d9d6587eee" />
---

### SVM Modeling — All Data vs. Female-Only vs. Male-Only

- **Goal**: Predict **ADHD** (and **Sex** in multi-output) with Support Vector Machines (SVM), comparing:
  - **All participants**
  - **Female-only subset**
  - **Male-only subset**

- **Preprocessing**:
  - One-hot encoded key categorical fields (ethnicity, race, parental education/occupation).
  - Dropped IDs/targets from features; filled missing values with training medians.
  - Standardized features with `StandardScaler` (fit on train).

- **Multi-Output SVM (ADHD + Sex) — Full Feature Set**:
  - Model: `MultiOutputClassifier(SVC(kernel='rbf', class_weight='balanced', probability=True))`
  - **Validation (ADHD)**: accuracy ≈ **0.61** (class-imbalance hurt recall for class 0).  
  - **Validation (Sex)**: accuracy ≈ **0.63** (better recall for class 0 than class 1).

- **Top-25 Shared Features (RF-derived) → Multi-Output SVM**:
  - Built a joint importance table from RF for ADHD & Sex, averaged importances, selected **25 features**.
  - **Validation (ADHD)**: accuracy ≈ **0.76**, **F1 ≈ 0.83** for ADHD positive — sizable lift over full feature set.  
  - **Validation (Sex)**: accuracy ≈ **0.67**, balanced performance across classes.
  - **Representative top features**: `SDQ_SDQ_Hyperactivity`, `SDQ_SDQ_Externalizing`, `SDQ_SDQ_Difficulties_Total`, `SDQ_SDQ_Internalizing`, plus several PCA-reduced connectome components (e.g., `43throw_147thcolumn`, `96throw_169thcolumn`).

- **Single-Output SVM (ADHD only)**:
  - Cross-validated SVM on standardized features.
  - **All data**: Acc ≈ **0.67** (CV), AUC ≈ **0.60** (CV).  
  - **Female-only**: Acc ≈ **0.59** (CV), AUC ≈ **0.53** (CV).  
  - **Male-only**: Acc ≈ **0.73** (CV), AUC ≈ **0.57** (CV).  

- **Single-Output SVM (with engineered/selected features)**:
  - On `final_model_data.csv` (engineered + selected features):  
    - **Val Acc ≈ 0.75**, **AUC ≈ 0.84**; precision/recall balanced (e.g., F1 ~ **0.79** for ADHD=1).  
    - Sex prediction with the same feature space reached near-perfect internal validation (indicates features strongly encode sex; treat with caution for overfitting).

- **Takeaways**:
  - **Feature selection matters**: restricting to RF-derived **Top-25** substantially improves ADHD performance in the multi-output setting.
  - **Sex-stratified data behave differently**: male-only models outperform female-only on raw features; engineered features help close the gap.
  - **Bias & leakage controls**: encoding fitted on combined schema, imputers/scalers fit **only** on training data; targets removed from test schema.

<img width="238" height="166" alt="Screenshot 2025-08-21 at 4 21 10 PM" src="https://github.com/user-attachments/assets/2cbb806c-233f-43b7-86fe-7f9551c71788" />

---

### Single-Task SVM on Engineered Feature Set (ADHD only)

- **Goal**: Train an ADHD classifier using the engineered/selected feature table `final_model_data.csv`  
  (mix of SDQ/APQ metadata + top PCA-based connectome components).

- **Data prep**:
  - Dropped identifiers + helper cols: `participant_id`, `Sex`, `Sex_F_y`.
  - Split: 80/20 train/validation (`random_state=42`).
  - Standardized features with `StandardScaler` (fit on train only).

- **Model**: `SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)`

- **Cross-validation (train fold, 5-fold)**:
  - **Accuracy**: **0.76 ± 0.07**
  - **ROC-AUC**: **0.84 ± 0.05**

- **Hold-out validation (20%)**:
  - **Accuracy**: **0.75**
  - **Class 0** (no ADHD): Precision **0.61**, Recall **0.73**, F1 **0.67**
  - **Class 1** (ADHD): Precision **0.84**, Recall **0.75**, F1 **0.79**

- **Subset analysis (same pipeline, stratified by sex)**:
  - **All data**: CV Acc **0.85 ± 0.03**, CV AUC **0.94 ± 0.03**; Val Acc **0.85**
  - **Female-only**: CV Acc **0.78 ± 0.18**, CV AUC **0.88 ± 0.16**; Val Acc **0.83**
  - **Male-only**: CV Acc **0.80 ± 0.06**, CV AUC **0.92 ± 0.06**; Val Acc **0.79**

- **Sex prediction sanity check (same feature table)**:
  - Internal validation reached **~100%** accuracy/AUC.  
    _Note_: indicates features encode sex very strongly → treat as a potential overfitting/shortcut risk; use careful external validation.

- **What improved performance?**
  - Using the **engineered feature set** (RF-selected metadata + PCA components) boosted ADHD classification vs. raw full feature space.
  - Standardization + class weighting helped with imbalance.

<img width="226" height="166" alt="Screenshot 2025-08-21 at 4 22 16 PM" src="https://github.com/user-attachments/assets/b2696a10-ebf8-4489-affa-35e8d4962411" />
---

### Multi-Output SVM (ADHD + Sex) on Engineered Feature Set

- **Goal**: Train a single model that jointly predicts **ADHD diagnosis** and **biological sex** using the curated feature table `final_model_data.csv`  
  (top PCA connectome components + key SDQ/APQ metadata).

- **Data prep**
  - Loaded `final_model_data.csv`, dropped `participant_id`, `Sex`, `Sex_F_y`.
  - **Features (`X`)**: all columns except labels; removed `Sex_F_x` from `X` to avoid target leakage in sex head.
  - **Targets (`y`)**: `ADHD_Outcome_x` (0/1), `Sex_F_x` (0=female, 1=male).
  - Split **80/20** train/validation (`random_state=42`); standardized with `StandardScaler` (fit on train only).

- **Model**
  - `MultiOutputClassifier(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))`
  - Trains two SVM heads under one wrapper: one for ADHD, one for Sex.

- **Validation results (hold-out 20%)**
  - **ADHD**:
    - Accuracy **0.75**
    - Class 0 (no ADHD): Precision **0.61**, Recall **0.73**, F1 **0.67**
    - Class 1 (ADHD): Precision **0.84**, Recall **0.75**, F1 **0.79**
  - **Sex**:
    - Accuracy **1.00**
    - Class 0 (female): P/R/F1 **1.00/1.00/1.00**
    - Class 1 (male):   P/R/F1 **1.00/1.00/1.00**

- **Takeaways**
  - The joint SVM maintains **strong ADHD performance** comparable to single-task SVM.
  - **Perfect sex prediction** suggests sex is highly encoded in the selected features; treat as a **shortcut/overfitting risk** and validate on external data or with stricter leakage checks.
---

### Tech Stack
- Python, Jupyter Notebook
- pandas, numpy, seaborn, matplotlib, plotly
- scikit-learn (SVM, RandomForest, PCA)

---


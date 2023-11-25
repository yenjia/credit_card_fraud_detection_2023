# Data Preprocessing Description

## Data Preparing
In this experiment, we have three tables to design the algorithm.
1. `training.csv`: The training table, which was provided in the beginning.
2. `public.csv`: The public test table with labels, which was provided on 11/20.
3. `private.csv`: The private test table without labels, which was also provided on 11/20.

## Data Preprocessing

This section involves handling missing values, encoding categorical data, and scaling numerical data.

### Categorical Data Preprocessing

The `categorical_preprocessing` function in this code performs the following tasks on categorical data:

1. Fills missing values with -1
2. Creates new features for each categorical variable:
   - `mode_group_{i}`: The most frequent category for each categorical variable within each "cano" group
   - `mode_group_{i}_2`: The most frequent category for each categorical variable within each "chid" group
   - `sum_cano_{i}`: The number of unique categories for each categorical variable within each "cano" group
   - `sum_chid_{i}`: The number of unique categories for each categorical variable within each "chid" group

### Numerical Data Preprocessing

The `numerical_preprocessing` function in this code performs the following tasks on numerical data:

1. Fills missing values with -1
2. Creates new features for each numerical variable:
   - `normd_group_{i}`: Normalized numerical value within each "cano" group
   - `mean_normd_group_{i}`: Mean of normalized numerical value within each "cano" group
   - `diff_normd_group_{i}`: Absolute difference between normalized numerical value and its mean within each "cano" group
   - `normd_group_{i}_2` Normalized numerical value within each "chid" group
   - `mean_normd_group_{i}_2`: Mean of normalized numerical value within each "chid" group
   - `diff_normd_group_{i}_2` Absolute difference between normalized numerical value and its mean within each "chid" group

### Overall Preprocessing

The code also performs the following preprocessing tasks:

* Counts the number of "cano" and "txkey" under the group "chid"
* Saves the preprocessed data to a CSV file

This preprocessing pipeline prepares the data for further training and inference.

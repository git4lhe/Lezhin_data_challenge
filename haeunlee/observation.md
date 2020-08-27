### [Observation]
- used Dask for large dataset
- found that large dataset transformation does not take a long time, however, the training time takes more than 10min.
- **THEREFORE**, decided to implement Dask for automated chunk computation
- works well with scikit-learn, numpy and pandas (convenient)

### Dask benchmarks
| nrows            	| cv 	| (best)param_n_estimators 	| split1_test_score 	| split2_test_score 	| mean_test_score   	|
|------------------	|----	|--------------------------	|-------------------	|-------------------	|-------------------	|
| 10000            	| 3  	| 200                      	| 0.785178517851785 	| 0.759375937593759 	| 0.771900075430458 	|
| all(approx 650k) 	| 3  	| 500                      	| 0.799520700768531 	| 0.79795058259648  	| 0.801600418698179 	|

### model comparison
- best score(to be updated) : random forest (approx 80% accuracy)
- worst score: lasso (approx 20% accuracy)

### [GOAL Todo list]
### data analysis
- correlation btw vars
  - delete var with high correlation (>0.8? )
- **DONE**
    - correlation features correlations over 0.8, found to be no correlation because of too many "nan values"
  
### preprocessing
- num: scaler
- cat: one-hot encoding
- delete 'ID' style var<br>
	+) ridge/lasso to see correlation btw vars
- PCA

- **DONE**
    - build basic pipeline for only num (in core/transforms.py)
    - categorical features considered confounder(혼란변수)
     
### models 
- basic results based 3 models (logistic regression must included)
- **DONE**
    - implemented lasso, SVM, Random Forest

### evaluation
- cross-validation stratified
- **DONE**
    - defined cv=3 (more than 3 takes too much time)
    - question: does large dataset needs cross validation?
    - as far as I know, cv is used for datasets that has few number of samples
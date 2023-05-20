**Machine Learning Landscape**
1. In general, what steps you need to take to process data before it can be 
        applied for building machine learning models (page 33)
2. Learn the language: `the general idea behind boosting is to transform weak 
        learners into strong learners by iteratively improving upon errors. 
        The key idea behind gradient boosting is to use gradient descent to 
        minimize the errors of the residuals`
3. Do you know the term `data wrangling`?
4. List ways to correcting null values in tabular data
5. Can you write down the sigmoid activation function?
6. What datasets are presented in this chapter?

---

**Decision Trees in Depth**
1. Can you explain the `Gini criterion` in decision trees, write down its
        formulation (page 75)
2. Can you describe model bias and variance (page 76) `contrasting 
        variance and bias`
3. Remember this language: `one of the best machine learning strategies 
        to strike a nice balance between variance and bias is to fine-tune 
        hyperparameters`
---

**Bagging with Random Forests**

1. Remember this line: `In machine learning, an ensemble method is a 
        machine learning model that aggregates the predictions 
        individual models. Since ensemble methods combine the results
        of multiple models, they are less prone to error, and therefore 
        tend to perform better`
2. Can you explain `Bootstrapping` and `Boosting`
3. Do you know the drawbacks of random forest
4. Can you explain `learning_rate` in the `GradientBoostingRegressor`? How 
        is is related to the `n_estimators` (page 139)
5. Do you know stochastic gradient boosting
6. Tell me about ensemble methods (bagging), boosting, and gradient boosting

---

**From Gradient Boosting to XGBoost**

1. can you tell me the difference between traditional gradient boosting 
        and XGBoost

---

**XGBoost Unveiled**

1. Do you know what did Tianqi Chen do to speed XGBoost? (page 152)
---

**XGBoost Hyperparameters**

1. about `scikit-learn`, do you know `cross_val_score`, `GridSearchCV`,         
        `RandomizedSearchCV`, and `StratifiedKFold`? (page 185)
2. can you explain why it's important not to become too invested in 
        obtaining the highest possible score? (page 187)
3. remember this: `learning_rate` shrinks the weights of trees for each 
        round of boosting. By lowering `learning_rate`, more trees are 
        required to produce better scores. Lowering `learning_rate` prevents 
        overfitting because the size of the weights carried forward is smaller
4. do you know early stopping in XGBoost? (page 194)


---

**Discovering Exoplanets with XGBoost: highly imbalanced dataset**

1. can you explain `precision`, `recall`, `f1 score`, and the difference 
        between `macro avg.` and `weighted avg.` (page 218)
2. do you know how to resample imbalanced data?
3. Q: can you name evaluation metrics in scikit-learn? A: from module 
        `sklearn.metrics` you can import `confusion_matrix`, 
        `classification_report`, `recall_score`, _etc._
4. Q: if the data is resampled before splitting it into training and test 
        sets, the recall score will the inflated. can you see why? 
        A: when resampling, nine copies will be made of the positive cases. 
        after splitting this data into training and test sets, copies are 
        likely contained in both sets. so, the test set will contain most of 
        the same data points as the training set. the appropriate strategy is 
        to split the data into a training and test set first and then to 
        resample the data 
5. Q: do you know how to use scikit-learn to tune hyperparameters? 
        A: yes, from module `scikit-learn.model_selection`, import 
        `GridSearchCV`, `RandomizedSearchCV`, 
        `StratifiedKFold`, and `cross_val_score` at first
6. Q: do you know how to finetune XGBoost? A: yes, `n_estimators`, 
        `learning_rate`, `max_depth`, `subsample`, `gamma` are the key 
        hyperparameters to tune. It's also worth trying `max_delta_step`, 
        which XGBoost only recommends for imbalanced datasets. As a final 
        strategy, combine `subasmple` with all the column samples in a 
        random search `colsample_bylevel`, `colsample_bynode`, 
        `colsample_bytree`
7. Q: if a ML model performs impressively on the training set and modestly 
        at best on the test set, the variance is likely to be low or high? 
        A: high

--- 

**XGBoost Alternative Base Learners**

1. Q: do you know what linear base learner of XGBoost? A: `gblinear`
2. Q: do you what's the difference between base learners 
        `gbtree` and `gblinear`? A: `gbtree` is for non-linear problem while 
        `gtlinear` is for linear problem. They have different hyperparameters
3. Q: do you know what is the `shotgun` and `coord_descent` updaters of 
        XGBoost? A: Coordinate descent is a machine learning term defined as 
        minimizing the error by finding the gradient one coordinate at a time 

---

**XGBoost Kaggle Masters**

1. remember this: `between 2014 and 2018, XGBoost consistently outperformed 
        the competition on tabular data -- data organized in rows and 
        columns as contrasted with unstructured data such as images or text, 
        where neural networks hnd a edge. With the emergence of LightGBM in 
        2017, a lighting-fast Microsoft version of gradient boosting, XGBoost 
        finally hand some real competition with tabular data.`
2. remember this: `implementing a great machine algorithm such as XGBoost or 
        LightGBM in Kaggle competitions isn't enough. Similarly, fine-tuning 
        a model's hyperparameters often isn't enough. While individual model 
        predictions are important, it's equally important to engineer new 
        data and to combine optimal models to attain higher scores.`
3. Q: can you list general approaches for validating and testing machine 
        learning models on your own? A: split data into a training set and 
        a hold-out set, split the training set into a training and test set 
        or use cross-validation, and then after obtaining a final model, test 
        it on the hold-out set
4. Q: do you know mean encoding or target encoding? A: Mean encoding transforms 
        categorical columns into numerical columns based on the mean target 
        variable. For instance, if the color orange led to seven target values 
        of 1 and three target values of 0, the mean encoded column would 
        be 7/10 = 0.7. Since there is data leakage while using the target 
        values, additional regularization techniques are required. For more 
        information on mean encoding, refer to this [Kaggle study](https://www.kaggle.com/code/vprokopev/mean-likelihood-encodings-a-comprehensive-study/notebook)
5. Q: what is data leakage? A: data leakage occurs when information between 
        training and test sets, or predictor and target columns, are shared. 
        the risk here is that the target column is being directly used to 
        influence the predictor columns, which is generally a bad idea in ML
6. Q: do you know non-correlated ensembles? A: using a majority rules 
        implementation, a prediction is only wrong if the majority of 
        classifiers get it wrong. It's desirable, therefore, to have a 
        diversity of models that score well but give different predictions. 
        if most models give the same predictions, the correlation is high, 
        and there is little value in adding the new model to the ensemble. 
        finding differences in predictions where a strong model may be wrong 
        gives the ensemble the chance to produce better results. Predictions 
        will be different when the models are non-correlated
7. Q: can you show the steps to find correlation between machine learning 
        models? A:
8. Q: do you know stacking models? how does it differ from (non-correlated) 
        ensemble? A: stacking combines machine learning models at two 
        different levels: the base level, whose models make predictions on 
        all the data, and the meta level, which takes the predictions of the 
        base models as input and uses them to generate final predictions 

---

**XGboost Model Deployment**

1. remember this: `deploying models for industry is a little different than 
        building models for research and competitions, in industry, automation 
        is important since new data arrives frequently. more emphasis is placed 
        on procedure, and less emphasis is placed on gaining minute percentage 
        points by tweaking machine learning models


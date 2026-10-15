Project StreamGuard: Predicting Subscriber Churn and Customer Lifetime Value
Introduction

Streaming platforms depend on subscriptions, and winning a new subscriber costs far more than keeping an existing one. When customers cancel, which is called churn, profit suffers. Project StreamGuard is an end-to-end machine learning pipeline that tackles this in two ways. It predicts which active subscribers are likely to cancel, which is a classification task. It also forecasts how much revenue each customer will bring in over their lifetime, which is a regression task. With both, a company can target retention offers and set acquisition budgets with some evidence.

Data

The project uses 600 synthetic subscriber profiles modelled on a Netflix-style service. Each profile records subscription type, preferred device, monthly fee, watch hours, logins, support interactions, tenure, whether the customer churned, and lifetime value. About 33% of the customers churned. The data was made messy on purpose, as real platform logs are, with missing values, a corrupted fee of $99,999, and negative watch hours.

Method

First, I cleaned the data. Impossible values were set to missing, numeric gaps were filled with the median, and categorical gaps with the most common value. Extreme values were found with the interquartile range (IQR) rule and capped rather than deleted, so no customer was lost.

Next, I engineered two behavioural features: an Engagement Efficiency Index (watch hours per dollar) and an Engagement Index combining watch hours and logins. Text categories were one-hot encoded and numeric features were standardized. The data was split 80/20 before any fitting, so the test set never influenced training.

I then used Recursive Feature Elimination with cross-validation, which kept 10 of 12 features. It dropped the two subscription-type columns, because the monthly fee already reveals the plan. All models were compared with stratified 5-fold cross-validation, so every fold had the same share of churners. I trained three baselines (Logistic Regression, K-Nearest Neighbors, Support Vector Machine) and two ensembles (Random Forest and AdaBoost), tuning the ensembles with randomized search.

Results

For churn prediction, Logistic Regression had the best cross-validated ROC-AUC at 0.756, ahead of Random Forest (0.744), SVM (0.743) and AdaBoost (0.734). On the 120 held-out customers it reached a ROC-AUC of 0.84, an accuracy of 77.5%, and a recall of 87%, catching 34 of the 39 customers who really churned. Recall mattered most, because missing a churner costs far more than offering a discount to someone who would have stayed.

For lifetime value, Gradient Boosting performed best. On the test set it had a mean absolute error of about $55, a root mean squared error of about $76, and an R² of 0.88, so the model explains about 88% of the variation in customer value.

Findings and Limitations

A surprising result was that the simple Logistic Regression model beat the tuned ensemble models on this dataset. A few cautions apply. The data is synthetic, so the scores show that the pipeline works rather than how it would do on real subscribers. The test set is small, so cross-validation scores are the more trustworthy estimate. The cleaning step also capped lifetime values at about $850, so the regression was scored against capped values. A real deployment should also tune the classification threshold to the actual cost of a retention offer versus a lost customer.

Conclusion

StreamGuard shows a complete, leak-free machine learning workflow, from messy raw data to saved, ready-to-use models. It also shows that a well-prepared dataset and a fair comparison matter more than choosing the most complex algorithm

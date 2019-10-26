# Analytics_Vidhya-Loan_Delinquency_Prediction
## Problem Statement-

Loan Delinquency Prediction is one of the most critical and crucial problem faced by financial institutions and organizations as it has a noteworthy effect on the profitability of these institutions. In recent years, there is a tremendous increase in the volume of non–performing loans which results in a jeopardizing effect on the growth of these institutions. Therefore, to maintain a healthy portfolio, the banks put stringent monitoring and evaluation measures in place to ensure timely repayment of loans by borrowers. Despite these measures, a major proportion of loans become delinquent. Delinquency occurs when a borrower misses a payment against his/her loan. Given the information like mortgage details, borrowers related details and payment details, our objective is to identify the delinquency status of loans for the next month given the delinquency status for the previous 12 months (in number of months). 

### Submissions are evaluated using F1-Score.

Versions of libraries used-

1)	Catboost  	: 0.14.2
2)	Scikit-learn 	: 0.20.1
3)	Xgboost  	: 0.80	
4)	Imblearn  	: 0.4.3
5)	Numpy 		: 1.15.4
6)	Pandas		: 0.24.2
7)	Matplotlib	: 3.0.2
8)	Seaborn	: 0.9.0
9)	Dateutils	: 2.8.0

## Approach-
   Since the data was extremely imbalanced, it was difficult for the model to learn and make prediction for the minority class. Therefore, I used SMOTE to artificially increase the instances of minority class and make it equal to the majority class. This oversampled data was used to train models.

## Data preprocessing:

1)	There were no missing values in any column. 
2)	Categorical data – Used pd.get_dummies  (One Hot encoding) to deal with categorical data like for loan purpose and source.
3)	Outlier removal - Removed outliers from credit score, loan to debt ratio, loan to value, unpaid principal, interest rate. 
4)	Oversampling – The loan default data was highly imbalanced i.e only about 0.5% data points had label 1. The minority class was oversampled using SMOTE (Synthetic Minority Oversampling Technique).

## Feature Engineering and feature selection:

1)	A new feature was created by taking difference of origination date and first payment date.
2)	 Features like insurance type, insurance percent, source _Z (that were created after get_dummies )  and financial institutions were dropped as they had very low correlation with the m13 label. When the model was trained while keeping these features it reduced the f1 score.
3)	All other features were kept as they had significant correlation with the m13 label.

## Final Model:

The final model is voting classifier of four models:
1)	Random Forest 1
2)	Random forest 2 (slightly different hyperparameters)
3)	Bagging classifier of Xgboost
4)	Bagging Classifier of Catboost.

I started experimenting with different models, out of which the individual random forest, xgboost and catboost were performing well on the validation set. Each model’s parameters were extensively tuned using gridsearchcv.  The final model was ensemble of all the good performing models to make predictions more robust. 

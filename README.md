# Kaggle-competitions

## Competition 1 - Titanic - Machine learning from Disaster
https://www.kaggle.com/c/titanic/submissions.  
Highest accuracy achieved in the competition - 77.9%

### Implemenation specifics:
Final features used for training & testing:    
1) PClass(Passenger class) - Converted to numeric categorical values  
2) Sex(Male/Female) - Converted to binary categorical values(0/1).  
3) Age - Converted to 4 age bands and each age band is assigned a numeric categorical values.  
4) Embarked - Converted to numeric categorical values.  
5) Family Size - Obtained by using initial input feature SibSp & Parch as SibSp + Parch + 1. This is again cut into band size of 3(small, medium and big family) and 
each band is converted to numerical categorical values.  

Now all the features are categorical and are directly fed to train the model

### Algorithm used
RandomForest classifier with random state = 0.  
Cross validation - 5-fold cross validation

### Installation
python3.  
pandas.  
scikit-learn.  


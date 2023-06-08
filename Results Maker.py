import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

def getBinResults():

    #Read csv files 
    trainData = pd.read_csv("TrainingDataBinary.csv", sep=",",header=None)
    testData = pd.read_csv("TestingDataBinary.csv", sep=",",header=None)


    #convert to dataframes
    df_train = pd.DataFrame(data = trainData)
    X_test=pd.DataFrame(data = testData)


    #Split up X and Y
    X_train=df_train.iloc[:,:-1]
    y_train=df_train.iloc[:,-1]

  
    #Scale Data

    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Train Extra Trees Classifier

    ETC = ExtraTreesClassifier( criterion= 'entropy' ,max_depth= None, max_features = None, min_samples_leaf = 2, min_samples_split= 2)
    ETC.fit(X_train, y_train)

    #Create predictions 

    preds = ETC.predict(X_test)

    #Write predictions to CSV

    pd.DataFrame(preds).to_csv('TestingResultsBinary.csv', index=False, header=False)


def getMultiResults():

    #Read csv files 
    trainData = pd.read_csv("TrainingDataMulti.csv", sep=",",header=None)
    testData = pd.read_csv("TestingDataMulti.csv", sep=",",header=None)


    #convert to dataframes
    df_train = pd.DataFrame(data = trainData)
    X_test=pd.DataFrame(data = testData)

    #Split up X and Y
    X_train=df_train.iloc[:,:-1]
    y_train=df_train.iloc[:,-1]

   

    #Scale Data

    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Train Extra Trees Classifier

    ETC = ExtraTreesClassifier( criterion= 'gini' ,max_depth=None, max_features = 'log2', min_samples_leaf = 1, min_samples_split=2)
    ETC.fit(X_train, y_train)

    #Create predictions 

    preds = ETC.predict(X_test)

    #Write predictions to CSV

    pd.DataFrame(preds).to_csv('TestingResultsMulti.csv', index=False, header=False)


if __name__ == "__main__" :
    getBinResults()
    getMultiResults()





    

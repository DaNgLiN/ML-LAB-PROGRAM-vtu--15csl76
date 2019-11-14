import pandas as pd 
msg=pd.read_csv('C:\\Users\\Gunjan\\Desktop\\ML_Alorithms-Day2\\NaiveBayesianClassifier\\data6.csv',names=['message','label']) #Tabular form data 
print('Total instances in the dataset:',msg.shape[0])

msg['labelnum']=msg.label.map({'pos':1,'neg':0}) 
X=msg.message
Y=msg.labelnum

print('\nThe message and its label of first 5 instances are listed below')
X5, Y5 = X[0:5], msg.label[0:5]
for x, y in zip(X5,Y5): 
    print(x,',',y)

# Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split 
xtrain,xtest,ytrain,ytest=train_test_split(X,Y) 
print('\nDataset is split into Training and Testing samples') 
print('Total training instances :', xtrain.shape[0]) 
print('Total testing instances :', xtest.shape[0])

# Output of count vectoriser is a sparse matrix
# CountVectorizer - stands for 'feature extraction'
from sklearn.feature_extraction.text import CountVectorizer 
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain) #Sparse matrix 
xtest_dtm = count_vect.transform(xtest)
print('\nTotal features extracted using CountVectorizer:',xtrain_dtm.shape[1])

print('\nFeatures for first 5 training instances are listed below') 
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names()) 
print(df[0:5])#tabular representation
#print(xtrain_dtm) #Same as above but sparse matrix representation

# Training Naive Bayes (NB) classifier on training data. 
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain) 
predicted = clf.predict(xtest_dtm)

print('\nClassstification results of testing samples are given below') 
for doc, p in zip(xtest, predicted):
    if p==1:
        pred = 'pos'  
    else:
        'neg'
        print('%s -> %s ' % (doc, pred))

#printing accuracy metrics 
from sklearn import metrics 
print('\nAccuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))

print('Recall :',metrics.recall_score(ytest,predicted), '\nPrecison :',metrics.precision_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))

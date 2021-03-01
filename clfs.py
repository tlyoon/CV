from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import LinearSVC
#from tensorflow import keras
#from sklearn.naive_bayes import CategoricalNB
#from sklearn.naive_bayes import ComplementNB
#from sklearn.decomposition import PCA


#### Note that for the sake of some historical reasons, the name of the 
#### classifiers has to carry a '()' suffix.

def ccl():
    classifiers=[
          BernoulliNB() ,\
          GaussianNB()  ,\
          Perceptron() ,\
          DecisionTreeClassifier()  ,\
          #'keras.Sequential'       ,\
          svm.SVC()                 ,\
          KNeighborsClassifier()    ,\
          svm.LinearSVC()           ,\
          BaggingClassifier()       ,\
          RandomForestClassifier()  ,\
          AdaBoostClassifier()      ,\
          LogisticRegression()      ,\
         ]
    return classifiers

# Notes:

#svm.SVC()                 ,\ ### only for binary classification
#svm.LinearSVC()           ,\ ### only for binary classification
#LogisticRegression()     ,\  ### only for binary classification 


# MultinomialNB()  ### does not work,  Negative values in data passed to MultinomialNB (input X)
# ComplementNB()   ### does not work,  Negative values in data passed to MultinomialNB (input X)
# CategoricalNB    ### does not work,  Negative values in data passed to MultinomialNB (input X)
classifiers=ccl()

from global_param import *
import pltcmtrx
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import clfs
import dictionary
#import keras
#import tensorflow as tf
import pickle

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

#import gen_dictionary
#from pathlib import Path

#Nround=1

###### begin def all_training(round,nephochs):
def all_training(round,nephochs):
    all_train_append=[]
    all_train_append=np.array(all_train_append)
    #dict_classes=gen_dictionary.gen_dictionary()
    dict_classes=dictionary.dict
    
    for i in range(len(clfs.ccl())):
        classifier=clfs.ccl()[i]
        print('classifier=',classifier)        
        for j in range(len(dict_classes)):
            targetclass=[ keys for keys in dict_classes ][j]
#            print('targetclass=',targetclass)
            Xdatafile=targetclass+'_ovsp.train.npy'
            ydatafile=targetclass+'_y_ovsp.train.npy'
                
            if classifier == 'keras.Sequential':
                ydatafile=targetclass+'_ynn_ovsp.train.npy'
#                print('see me ?, classifier?',classifier)
            
            temp=training(Xdatafile,ydatafile,classifier,round,nephochs)
            all_train_append=np.append(all_train_append,[temp])
#            with open('trained_models_performance.txt', 'a+') as outfile: 
            #with open('trained_models_performance.npy', 'a+') as f:
            #    np.save(f,temp)             
            #p = Path('trained_models_performance.npy')
            #with p.open('ab') as f:
            #    np.save(f, temp)               
            #print('all_train_append = ',all_train_append)
    return all_train_append
            #print(Xdatafile,ydatafile)
###### end def all_training(round,nephochs): ##################################

    
###### begin def training(Xdatafile,ydatafile,classifier,round,nephochs): #####
def training(Xdatafile,ydatafile,classifier,round,nephochs):
 
    
    #from numpy import mean
    #from sklearn.model_selection import cross_val_score
    #from sklearn.model_selection import RepeatedStratifiedKFold
    
    if not os.path.isdir('saved_model'):
        os.mkdir('saved_model')
   
            
    ##### specify input data files
    X=np.load(Xdatafile)
    y=np.load(ydatafile)
    ##### end of specify input data files
    lenclass=2#len(set(y))
#    print('training_1vaA_ec for '+ str(classifier) + 'is initiated')
    print('training_1vaA for '+ str(classifier) + 'is initiated')
    print('Xdatafile is ', Xdatafile)
    print('ydatafile is ', ydatafile)
    print('number of distinct classes to be classified =',lenclass)

    if classifier != 'keras.Sequential':
        ### reshape X
        X=np.reshape(X,(X.shape[0],X.shape[1]*X.shape[1]))
        ### splitting data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42,test_size=0.5)
        ### end of splitting data     
################ for non-nn model ##############################        
        if  isinstance(classifier, RandomForestClassifier) \
            or isinstance(classifier, AdaBoostClassifier) \
            or isinstance(classifier, BaggingClassifier) \
            or isinstance(classifier, KNeighborsClassifier):
                
            print('classifier = ',classifier,'To invoke GS ') 
            #create a dictionary of all values we want to test for n_estimators
            params_space = {'n_estimators': [50, 100, 200]}#use gridsearch to test all values for n_estimators
            
            if isinstance(classifier, KNeighborsClassifier):
                params_space={'n_neighbors': np.arange(1, 25)} #use gridsearch to test all values for n_neighbors
    
            classifier_gs = GridSearchCV(classifier, params_space, cv=5,n_jobs=-1 )#fit model to training data
            #classifier_gs = GridSearchCV(classifier, params_space, cv=lenclass,n_jobs=-1 )#fit model to training data
            classifier_gs.fit(X_train, y_train)
            #save best model
            classifier_best = classifier_gs.best_estimator_ #check best n_estimators value
            classifier=classifier_best
    #        print('classifier_gs.best_estimator= ',classifier_gs.best_estimator_)#check best n_estimators value
    #        print('classifier_gs.best_params_=',classifier_gs.best_params_)
        else:
            print('classifier = ',classifier,'Not to invoke GS ') 
            
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
################ end for non-nn model ##############################

################ for nn model ##############################        
    elif classifier == 'keras.Sequential':
        print('nn classifier = ',classifier)
        ### note that no data reshaping for X is required for nn
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42,test_size=0.5)
        img_rows, img_cols = np.shape(X_train)[1], np.shape(X_train)[2]
        model = keras.Sequential([
        keras.layers.Flatten(input_shape=(img_rows, img_cols)),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
        ])
        print('model.summary:')        
        model.summary(line_length=None, positions=None, print_fn=None)
        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=nephochs)
        y_pred = model.predict(X_test)
        #print('y_test:')
        y_pred=np.round(np.array(y_pred)).astype(int)
        #print(y_pred)
    
################ end for nn model ##############################     
       
    #nocsf=str(classifier).split('(')[0]
    targetclass=Xdatafile.split('_ovsp.train.npy')[0]
       
    ############# for non nn classifier 
    if classifier != 'keras.Sequential':
        nocsf=str(classifier).split('(')[0]+'.'+ targetclass
        # cross-validatation
        # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
        nfold=2 ### default  10. Minimum 2.
        n_repeats=1   #### default 10
        print(str(nfold)+'-fold cross validation, repeated for '+str(n_repeats)+' times')
        cv = RepeatedStratifiedKFold(n_splits=nfold, n_repeats=n_repeats, random_state=None)
        scores = cross_val_score(classifier, X, y, cv=cv, n_jobs=-1)
        # scoring='roc_auc'
        print('scores:', scores)
        print('mean of scores: %.3f' % mean(scores))
    #### confusion matrix 
        cm = confusion_matrix(y_test, y_pred)
    #    fig, ax = plt.subplots()
    #    pltcmtrx.plot_confusion_matrix(cm, classes=np.unique(y), title=str(classifier))
    ### end of function to plot confusion matrix 
        print('confusion matrix:')
        print(cm)
    
        #### SAVE trained model using pickle ###########################
        
    #    modelfile='saved_model/' + str(round)+'.'+nocsf + '_' + str(lenclass) + '.pkl'
        modelfile='saved_model/' + str(round)+'.'+ nocsf + '.pkl'
        with open(modelfile,'wb') as f:
            pickle.dump(classifier,f)
            
        mas=metrics.accuracy_score(y_test, y_pred)
        mas='{:5.3f}'.format(mas)
        gm=geometric_mean_score(y_test, y_pred)
        gm='{:5.3f}'.format(gm)
        #    tt='evaluate'+str(j)+'.'+str(classifier) +'; a_scr='+mas+'; geo_mean='+gm
        tt='training'+str(round)+'.'+nocsf  +'; a_scr='+mas+'; geo_mean='+gm
        #pltcmtrx.plot_confusion_matrix(cm, classes=np.unique(y), title=tt)
        #plt.show()
        print('modelfile=',modelfile)
        print("classifier=",classifier)
        print('classification_report(y_test, y_pred):')
        print(classification_report(y_test, y_pred))
        print('')
        try:
            classifier=str(classifier).split('(')[0]
        except:
            dummy=0
#        print('00000000, classifier:',classifier)    
        return \
            [ \
            modelfile, \
            Xdatafile, \
            classifier, \
            mas, \
            gm, \
            cm, \
            classification_report(y_test, y_pred), \
            scores, \
            mean(scores)\
            ]       
    ############# end for non nn classifier
    
    
    ############# for nn classifier
    if classifier == 'keras.Sequential':
        nocsf=str(classifier)+'.'+ targetclass
        modelfile='saved_model/' + str(round)+'.'+ nocsf + '.pkl'
        #print('before')
        model.save(modelfile)
        #print('after')
        print('modelfile=',modelfile)
        print("classifier=",classifier)
        cm=np.zeros(shape=(2,2))
        y_pred=np.argmax(y_pred, axis=1)
        y_test=np.argmax(y_test, axis=1)
        cr=classification_report(y_test, y_pred)
        print("classification_report(y_test, y_pred)",classification_report(y_test, y_pred))
        cm=confusion_matrix(y_test, y_pred)
        print('cm:')
        print(cm)
        #cv = RepeatedStratifiedKFold(n_splits=nfold, n_repeats=n_repeats, random_state=None)
        #scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
        mas=metrics.accuracy_score(y_test, y_pred)
        print('accuracy score=','{:5.3f}'.format(mas))
                        
        scores=mas
        print('scores:', '{:5.3f}'.format(scores))

        ms=mas  #mean(scores)
        print('mean of scores',ms)
        
        gm=geometric_mean_score(y_test, y_pred)
        gm='{:5.3f}'.format(gm)
        print('gm=',gm)
        
        tt='training'+str(round)+'.'+nocsf  +'; a_scr='+str(mas)+'; geo_mean='+str(gm)
        #pltcmtrx.plot_confusion_matrix(cm, classes=np.unique(y), title=tt)
        print(cr)
        return \
            [ \
            modelfile, \
            Xdatafile, \
            classifier, \
            mas, \
            gm, \
            cm, \
            cr, \
            scores, \
            ms 
            ]       
    print('round =',round)
    print(modelfile,'saved')
                                
###### end def training(Xdatafile,ydatafile,classifier,round,nephochs): #####
    

###### begin def exe_train_rounds(Nround,nephochs): ############
#def exe_train_rounds(Nnofs,Nnofs_evaluate,fractrain, nephochs,Nround):
def exe_train_rounds(Nnofs_evaluate,fractrain, nephochs,iNround_evaluate):
    #Nround=1
#def exe_train_rounds():
    #import data_preprocess_1vA
    #import clfs
    
    classifiers=clfs.classifiers
    classifiers=[ classifiers[i] for i in range(len(classifiers)) ]
    
    #dict_classes=gen_dictionary.gen_dictionary()
    dict_classes=dictionary.dict
    classes = [ keys for keys in dict_classes ]
    #    print('classes --- ',classes)
    #### initialise dictionary
    #dict_perf={};dict_perf2={};
    dict_csbfcl={}
    dict_csbfclk={}
    #### end of initialise dictionary
    
#    for round in range(Nround):
    for round in range(iNround_evaluate,iNround_evaluate+1):    
    #for round in range(1):
        all_train_append=all_training(round,nephochs)
        lx=int(all_train_append.shape[0]/9)
        ly=9
        all_train_append=all_train_append.reshape(lx,ly)
        print('round = ',round)
        with open("train_performance.txt", "a") as a_file:
                a_file.write(
                         'round'+' classifier'+' class'+' current_score'
                         +'\n')                    
                
#        with open("train_performance.txt", "a") as a_file:
#            a_file.write("round="+str(round)+'\n')

            
        for k in classes:
            print('The mean score of discriminating',k,'from other classes by various classifiers:')
            #with open("train_performance.txt", "a") as a_file:
 #               a_file.write('The mean score of discriminating '+k+' from other classes by various classifiers:'+'\n')
            #    a_file.write(
            #             'round'+' classifier'+' class'+' current_score'
            #             +'\n')                    
                
            for i in range(all_train_append.shape[0]):
        #        for j in range(all_train_append.shape[1]):   
                if all_train_append[i][1].split('_')[0]==k:
                   #statement='class='+str(all_train_append[i][1].split('_')[0]) \
        #+' classifier='+str(all_train_append[i][2]) \
        #+' mean score='+str(all_train_append[i][8])          
                   statement=str(all_train_append[i][2])+' mean score: '+str(all_train_append[i][8])
                   print(statement) 
                   print('=====================================')
                   print('all_train_append[i][2])',str(all_train_append[i][2]))
                   #pklname=str(all_train_append[i][2])+'.'+k
                   #print('pklname::::',pklname)
                   #pklname2=str(round)+'.'+pklname1+'.pkl'
                   current_score=all_train_append[i][8]
                   print('round','classifier','class','score')
                   print(round,str(all_train_append[i][2]),k,current_score)
                   dict_csbfclk[round,str(all_train_append[i][2]),str(k)] = current_score
                   #print('&&&&&&&& dict_csbfclk:',dict_csbfclk)
                   with open("train_performance.txt", "a") as a_file:
                     a_file.write(
                         str(round)+' '+ str(all_train_append[i][2])+' '+k+' '+str(current_score)+' '+sampler_train
                         +'\n')
                   
                   #if current_score >= dict_perf[pklname]:
                   #    rmax=round
                   #    accmax=current_score
                   #    dict_perf[pklname]=rmax
                   #    dict_perf2[pklname]=accmax
                   
                   #print(round,all_train_append[i][2],k,all_train_append[i][8])
                   #with open("train_performance.txt", "a") as a_file:
                   #  a_file.write(statement+'\n')
        #            print('i=',i)
        #            print('name of the trained model = ',all_train_append[i][j])
        #            print('class = ',all_train_append[i][1].split('_')[0])          
        #            print('classifier = ',all_train_append[i][2])          
        #            print('metrics.accuracy_score = ',all_train_append[i][3])
        #            print('geometric_mean_score = ',all_train_append[i][4])
        #             print('confusion matrix:',all_train_append[i][5])
        #            print('classification_report:',all_train_append[i][6])
        #            print('scores of k-fold verification:',all_train_append[i][7])
        #            print('mean score:',all_train_append[i][8])
            print(' ')
            with open("train_performance.txt", "a") as a_file:
                a_file.write('\n')
        #subprocess.call('python data_preprocess_1vA.py')
        #import shlex
        #shell_cmd='python data_preprocess_1vA.py'
        #subprocess_cmd = shlex.split(shell_cmd)
        #subprocess.call(subprocess_cmd)
        #
        #if round < Nround-1:
        #    data_preprocess_1vA.all_target_training_data(Nnofs,Nnofs_evaluate,fractrain)
                                                        
    
    ### abstract class-specific best performing classifiers
    for i in classifiers:
        try:
            i=str(i).split('(')[0]
        except:
            dummy=0            
        print('')
        for j in classes:
#            print('i,j,Nround-1,dict_csbfclk:',i,j,Nround-1,dict_csbfclk)
#            arr= [ dict_csbfclk[r,str(i),j] for r in range(Nround) ]
            arr= [ dict_csbfclk[r,str(i),j] for r in range(iNround_evaluate,iNround_evaluate+1) ]
#            print('arr=',arr)
#            print('max=',max(arr))
            pos=arr.index(max(arr))
#            print('post=',pos)
            try:
                ii=str(i).split("()")[0]+str(i).split("()")[1]
            except:
                ii=str(i)
            best_perform_clf=str(pos)+'.'+ ii +'.'+ j +'.pkl'
            dict_csbfcl[ii,j]=best_perform_clf
    
#    print("class-specific best-performing classifier,'dict_csbfcl':")
    #print(dict_csbfcl)
#    for i in dict_csbfcl:
#        print(dict_csbfcl[i])
    
#    savedict = open('dict_csbfcl.py', 'w')
#    savedict.write('dict_csbfcl='+ str(dict_csbfcl))
#    savedict.close()
    ### end abstract class-specific best performing classifiers
#    return  dict_csbfcl
###### end def exe_train_rounds(Nround,nephochs): ############
   
iNround_evaluate=np.load('iNround_evaluate.npy')
iNround_evaluate=iNround_evaluate.item()
exe_train_rounds(Nnofs_evaluate,fractrain, nephochs,iNround_evaluate)

#Nround=1
#dict_csbfcl=exe_train_rounds(Nnofs,Nnofs_evaluate,fractrain, nephochs,Nround)

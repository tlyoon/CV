import gen_dictionary
import os,itertools
import numpy as np
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN #
from sklearn.preprocessing import normalize
from numpy import where
import imblearn
from imblearn.over_sampling import KMeansSMOTE
from global_param import *
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE

#print("Nnofs,fractrain,Nround,Nnofs_evaluate:",{Nnofs,fractrain,Nround,Nnofs_evaluate})

def target_training_data(targetclass):
    ##### target_training_data((targetclass)) is meant for genearing second half for training data set
    ##### generation of evalation data set in outsourced to all_target_training_data(Nnofs,Nnofs_evaluate,fractrain):
    
    import dictionary
    dictionary=dictionary.dict
    print(' ')
#    print('working on training set:')
#    print('targetclass=',targetclass)
    print('Resampling training set for class', targetclass)
    X1=3;X2=40  ##### components in the high-dimensional data point to be displayed for visualisation
    #dict_classes=gen_dictionary.gen_dictionary()
    #dict_classes=dictionary
    classes=[ keys for keys in dictionary ]
    classdir=classes
    traincontainer=[];traincontainer_y=[]
    origcontainer_y=[];origcontainer=[]
    origcontainer_ynn=[];
    traincontainer_ynn=[]
    appendsecondhalf=[]
    lensh=0
    for i in range(len(classdir)):
        cl=classdir[i]
        dirinclass=os.listdir(cl)
        lendirinclass=len(dirinclass)   
        dirinclass=[ os.path.join(cl,dirinclass[i],'ta.npy') for i in range(lendirinclass) ]  
        #print('i=',i,'cl=',cl, 'lendirinclass=',lendirinclass)
        
    #################### targetclass ############
        fnorig=str(targetclass)+'.orig.npy'
        fnorig_y=str(targetclass)+'_y.orig.npy'
        fnorig_ynn=str(targetclass)+'_ynn.orig.npy'
        fntrain=str(targetclass)+'.train.npy'
        fntrain_y=str(targetclass)+'_y.train.npy'
        fntrain_ynn=str(targetclass)+'_ynn.train.npy'
        
        if cl==targetclass:
#            print('{i, cl }=',{i,cl})
#            print('in target class',targetclass)
            #print('dirinclass=',dirinclass)        
            shuffle(dirinclass)
            firsthalf=dirinclass[0:int(fractrain*len(dirinclass))]
            secondhalf=dirinclass[int(fractrain*len(dirinclass)):]
            appendsecondhalf.append(secondhalf)
#            print('firsthalf=',firsthalf)
#            print('secondhalf=',secondhalf)       
            #####################################
            for k in range(len(firsthalf)):
             #       print('to append', firsthalf[k],'into',fnorig)
                    datain=np.load(firsthalf[k])
                    origcontainer.append(datain)
             #       print('to append', 0,'into',fnorig_y)
                    origcontainer_y.append(0)
                    origcontainer_ynn.append([0,1])
 #                   print('print from firsthalf:')
 #                   print('k:',k,'targetclass:',targetclass,'dictionary[targetclass]:',dictionary[targetclass])
    
            for k in range(len(secondhalf)):
              #      print('to append', secondhalf[k],'into',fntrain)
                    datain=np.load(secondhalf[k])
                    traincontainer.append(datain)
               #     print('to append',0,'into',fntrain_y)
                    traincontainer_y.append(0)
                    traincontainer_ynn.append([0,1])
            #####################################            
                    
        else:
#            print('{i, cl }=',{i,cl})
#            print('classes other than targetclass',targetclass)
    
            for k in range(len(dirinclass)):
                nnpy=dirinclass[k]
                if os.path.isfile(nnpy):
                    datain=np.load(nnpy)
                #    print('to append', nnpy,'into',fntrain)
                    traincontainer.append(datain)
                 #   print('to append', '1','into',fntrain_y)
                    traincontainer_y.append(1)
                    traincontainer_ynn.append([0,1])    
    
    origcontainer=np.array(origcontainer)
    origcontainer_y=np.array(origcontainer_y)
    
    traincontainer=np.array(traincontainer)
    traincontainer_y=np.array(traincontainer_y)
    
    origcontainer_ynn=np.array(origcontainer_ynn)
    traincontainer_ynn=np.array(traincontainer_ynn)
    
#    np.save(fnorig, origcontainer)                            
#    np.save(fnorig_y,origcontainer_y)
#    np.save(fnorig_ynn,origcontainer_ynn)
    np.save(fntrain, traincontainer)                            
    np.save(fntrain_y,traincontainer_y)
    np.save(fntrain_ynn,traincontainer_ynn)
    #################### end of targetclass ############
        
    ####################################################
    #print('Begin of oversampling for trainning set ')
    #k=len(classdir);
    k=2;
    seed=10
    X = traincontainer
    y = traincontainer_y
    #ynn = traincontainer_ynn
    
    #print('1',X.shape)
    X = np.reshape(X, (X.shape[0], X.shape[2]*X.shape[2]))
    #print('2',X.shape)
    
    ####### scatter plot of X and y
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.scatter(X[:, X1], X[:, X2], marker='o', 
    #               c=y, s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    #plt.show()
    
    #### creating sampling_strategy #####
    #lensh=max(lensh,len(secondhalf))
    #print('maxlensh  ======== ',lensh)
    sampling_strategy={}
    #sampling_strategy[0]=Nnofs*list(y).count(0)  
    #sampling_strategy[0]=Nnofs*list(y).count(1)  
    #sampling_strategy[1]=Nnofs*list(y).count(1)  
    print('npycountt:',npycountt)
    sampling_strategy[0]=Nnofs*npycountt
    sampling_strategy[1]=Nnofs*npycountt

    print('sampling_strategy (training set) = ',sampling_strategy)
    
  #  print("counter before oversampling = ", sorted(Counter(y).items()))
    
    
    ##### implementing oversampling ####
    if sampler_train == 'SMOTE':
        k=2;seed=10;n_jobs=-1;
        X_res, y_res= SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k-1, random_state=seed,n_jobs=n_jobs)\
                      .fit_resample(X, y)

    if sampler_train == 'BorderlineSMOTE':
        k=2;seed=10;n_jobs=-1;
        X_res, y_res=imblearn.over_sampling.BorderlineSMOTE(sampling_strategy=sampling_strategy,random_state=seed,k_neighbors=k,n_jobs=n_jobs) \
        .fit_resample(X, y)

    if sampler_train == 'ADASYN':
        k=3;seed=10;n_jobs=-1;
        X_res, y_res = ADASYN(random_state=seed,sampling_strategy=sampling_strategy,n_neighbors=k+1,n_jobs=n_jobs)\
            .fit_resample(X, y)
    
    if sampler_train == 'KMeansSMOTE':
        k=2;seed=10;n_jobs=-1;
        X_res, y_res = KMeansSMOTE(sampling_strategy=sampling_strategy,random_state=seed,k_neighbors=k+2,n_jobs=n_jobs)\
            .fit_resample(X, y)
            
    if sampler_train == 'RandomOverSampler':
        k=2;seed=10
        X_res, y_res = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=seed)\
            .fit_resample(X, y)
            
    if sampler_train == 'SVMSMOTE':
        k=4
        m_neighbors=2*k
        n_jobs=-1;seed=10;
        X_res, y_res = SVMSMOTE(sampling_strategy=sampling_strategy,random_state=seed,k_neighbors=k,n_jobs=n_jobs)\
            .fit_resample(X, y)
    
    
    #### implementing oversampling ####
    y_resnn = [ [y_res[i], np.abs((y_res[i]**(1) - 1))] for i in range(len(y_res))]    
    
    
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.scatter(X_res[:, X1], X_res[:, X2], marker='o', 
    #               c=y_res, s=25, edgecolor='k', cmap=plt.cm.coolwarm)
    #plt.show()
 #   print("counter before oversampling (trainning set) = ", sorted(Counter(y).items()))
 #   print("counter after oversampling (trainning set) = ", sorted(Counter(y_res).items()))
    dim=int(X_res.shape[1]**0.5)
    X_res=X_res.reshape(X_res.shape[0],dim,dim)
       
    ### report sizes of data before and after oversampling
#    norig=sum([ Counter(y)[keys] for keys in Counter(y) ])
#    print('Total number of data before oversampling:',norig)
#    novsp=sum([ Counter(y_res)[keys] for keys in Counter(y_res) ])
#    print('Total number of data after oversampling:', novsp)
#    print('Ratio of number of data after and before oversampling of trainning data:',novsp/norig)
### here     

    print("counter before oversampling (trainning set) = ", sorted(Counter(y).items())[0])
    norigsample=sorted(Counter(y).items())[0][1]
    print("counter after oversampling (trainning set) = ", sorted(Counter(y_res).items()))
    norig=sum([ Counter(y)[keys] for keys in Counter(y) ])
    print('Total number of data before oversampling:',norigsample)
    #novsp=sum([ Counter(y_res)[keys] for keys in Counter(y_res) ])
    novsp=Counter(y_res)[0]
    print('Total number of data after oversampling:', novsp)
    print('Ratio of number of data after and before oversampling of trainning data:',novsp/norigsample)

### end here
    ### save oversampled data
    fnres=targetclass+'_ovsp.train.npy'
    fnres_y=targetclass + '_y_ovsp.train.npy'
    fnres_ynn=targetclass + '_ynn_ovsp.train.npy'
    np.save(fnres_y, y_res)
    np.save(fnres_ynn, y_resnn)
    np.save(fnres, X_res)
    ####################################################
    return firsthalf,appendsecondhalf
#### end of def target_training_data(targetclass):

    
def all_target_training_data(Nnofs,Nnofs_evaluate,fractrain):
    dict_classes=gen_dictionary.gen_dictionary()
    classes=[ keys for keys in dict_classes ]
    
    import dictionary
    dict2=dictionary.dict2
    
    append_firsthalf=[]
    secondhalf=[]
    
    print('Begin of oversampling for trainning set ')
    print('imbalanced-learn samplers:',sampler_train)
    for targetclass in classes:
        call_target_training_data=target_training_data(targetclass) #### the call to target_training_data(targetclass) will generate second half as well 
        firsthalf=call_target_training_data[0]  
        append_firsthalf.append(firsthalf)
        
        secondhalfclass=call_target_training_data[1]
        secondhalf.append(secondhalfclass)        

#    all_orig_files_sh=list(itertools.chain.from_iterable(secondhalf))
#    print(' ')
#    print('orig data files from second half  (for trainning purpose):', \
#          list(itertools.chain.from_iterable(secondhalf)))
#    print(' ')
    

# =============================================================================
# ###### treating first half ###########################################################
#     #print(" ****************************** ")
#     all_orig_files_fh=list(itertools.chain.from_iterable(append_firsthalf))
#     #print('orig_data files from first half (for evaluation purpose):',all_orig_files_fh)
#     dictionary=dict_classes
#     all_orig_classes_data=[]
#     all_orig_classes_data_y=[]
#     for i in all_orig_files_fh:
#         datain=np.load(i)
#         all_orig_classes_data.append(datain)
#         clas=i.split(os.sep)[0]
#         all_orig_classes_data_y.append(dictionary[clas])
#         #print(i,clas,dictionary[clas])
#     np.save('all_orig_classes_data_y.npy',all_orig_classes_data_y)# to be oversampled
#     np.save('all_orig_classes_data.npy',all_orig_classes_data)# to be oversampled   
#     
#     all_orig_classes_data=np.array(all_orig_classes_data)
#     all_orig_classes_data=all_orig_classes_data.reshape(all_orig_classes_data.shape[0], \
#                                   all_orig_classes_data.shape[1]*all_orig_classes_data.shape[2])
#     #print('all_orig_classes_data.shape',all_orig_classes_data.shape)
#     print("")
# =============================================================================

# =============================================================================
# #### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     print('Begin of oversampling for evaluation set')
#     print('imbalanced-learn samplers:',sampler_eval)
#     
#     y=all_orig_classes_data_y
#     X=all_orig_classes_data
#  #### creating sampling_strategy #####
#     nlargest=max(Counter(y).values())
#     ovspnumber=Nnofs_evaluate*nlargest
#     lenclasses=len(classes)
#     sampling_strategy={}
#     
#     for i in range(lenclasses):
#         sampling_strategy[i] = ovspnumber
#         
#     #sampling_strategy[0]=Nnofs*list(y).count(0)  
#     #sampling_strategy[0]=Nnofs*list(y).count(1)  
#     #sampling_strategy[1]=Nnofs*list(y).count(1)  
#     print('sampling_strategy for evaluation data = ',sampling_strategy)
#     
#     ##### implementing oversampling ####
#     
#     ##### implementing oversampling ####
#     if sampler_train == 'SMOTE':
#         k=2;seed=10;n_jobs=-1;
#         X_res, y_res= SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k-1, random_state=seed,n_jobs=n_jobs)\
#                       .fit_resample(X, y)
# 
#     if sampler_train == 'BorderlineSMOTE':
#         k=2;seed=10;n_jobs=-1;
#         X_res, y_res=imblearn.over_sampling.BorderlineSMOTE(sampling_strategy=sampling_strategy,random_state=seed,k_neighbors=k,n_jobs=n_jobs) \
#         .fit_resample(X, y)
# 
#     if sampler_train == 'ADASYN':
#         k=3;seed=10;n_jobs=-1;
#         X_res, y_res = ADASYN(random_state=seed,sampling_strategy=sampling_strategy,n_neighbors=k+1,n_jobs=n_jobs)\
#             .fit_resample(X, y)
#     
#     if sampler_train == 'KMeansSMOTE':
#         k=2;seed=10;n_jobs=-1;
#         X_res, y_res = KMeansSMOTE(sampling_strategy=sampling_strategy,random_state=seed,k_neighbors=k+2,n_jobs=n_jobs)\
#             .fit_resample(X, y)
#             
#     if sampler_train == 'RandomOverSampler':
#         k=2;seed=10
#         X_res, y_res = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=seed)\
#             .fit_resample(X, y)
#             
#     if sampler_train == 'SVMSMOTE':
#         k=4
#         m_neighbors=2*k
#         n_jobs=-1;seed=10;
#         X_res, y_res = SVMSMOTE(sampling_strategy=sampling_strategy,random_state=seed,k_neighbors=k,n_jobs=n_jobs)\
#             .fit_resample(X, y)
#     
#     ##### end of implementing oversampling ####        
#     
#     dict_y={}
#     for i in range(lenclasses):
#         dict_y[i]=[0]*(lenclasses)
#         dict_y[i][i]=1
#     #print(dict_y)
#     y_resnn=np.empty((len(y_res),lenclasses),dtype=int)
#     
#     for j in range(lenclasses):
#         fnappend=[];fnyappend=[];fnynnappend=[]
#         print('')
#         for i in range(len(y_res)):
#             y_resnn[i]=dict_y[y_res[i]]
#         
#             if y_res[i]==j:
#                 
# #                print('i,y_res[i],y_resnn[i],j:',i,y_res[i],y_resnn[i],j)
# #                print('i,j:',i,j)
# 
#                 fn=dict2[j]
#                 fnname=fn+'_ovsp.eval.npy'
# #                print('To append X_res[i] data number',i,'with class index',j,'into',fnname)          
#                 fnappend.append(X_res[i])
#                 
#                 #fny=dict2[j]
#                 fnyname=fn+'_y_ovsp.eval.npy'
# #                print('To append',0,'with class index',j,'into',fnyname)
#                 fnyappend.append(0)
#                 
#                 #fnynn=dict2[j]
#                 fnynnname=fn+'_ynn_ovsp.eval.npy'     
# #                print('To append [0,1] with class index',j,'into',fnynnname)
#                 fnynnappend.append([0,1])
#         np.save(fnname,fnappend)
#         np.save(fnyname,fnyappend)
#         np.save(fnynnname,fnynnappend)
#         
# #        print('j=',j,'fnname,fnyname,fnynname:',fnname,fnyname,fnynnname)
#         print("Resampling evalatuation set for class",dict2[j])
#         print("counter before oversampling (evaluation set) = ", sorted(Counter(y).items()))
#         print("counter after oversampling (evaluation set) = ", sorted(Counter(y_res).items()))
#         norig=sum([ Counter(y)[keys] for keys in Counter(y) ])
#         print('Total number of data before oversampling:',norig)
#         novsp=sum([ Counter(y_res)[keys] for keys in Counter(y_res) ])
#         print('Total number of data after oversampling:', novsp)
#         print('Ratio of number of data after and before oversampling for evaluation data:',novsp/norig)
#         print('end of oversampling for evaluation set')    
#         print('')
# =============================================================================
#    print('I should see an end')            
    ###################################################
# end of def all_target_training_data(Nnofs,fractrain):    
 
### uncomment the following if you want to run this file per se 


state="rm -rf dat; for i in $(ls -d training/*) ; do ls $i | awk 'END{print NR}' >> dat; cat dat | awk '{s+=$1} END {print s}'; done |  awk 'END {print $1 }' > npycount.dat; rm dat;"

#state="for i in $(ls -d training/*) ; do ls $i | awk 'END{print NR}' >> dat; cat dat | sort -nrk1,1 dat | head -1 | cut -d ' ' -f3 ; done |  awk 'END {print $1 }' > npycount.dat; rm dat"

os.system(state)
with open('npycount.dat') as f:
    npycountt=int(f.read().splitlines()[0])
print('npycountt:',npycountt,type(npycountt))
os.remove('npycount.dat')

all_target_training_data(Nnofs,Nnofs_evaluate,fractrain)

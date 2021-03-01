import os
import numpy as np
import pickle

#origdir=os.path.basename(os.getcwd())
origdir=os.getcwd()
print('origdir:',origdir)
print('')

inference_dir="inference"
inference_dir=os.path.join(origdir,inference_dir)
print('inference_dir:',inference_dir)


votingclass=list(set([ item.split('.')[0] for item in os.listdir(inference_dir) if item.split('.')[-1]=='npy' ]))
print('votingclass:',votingclass)

os.chdir("saved_model")
dir_sm=os.getcwd()
print('dir_sm:',dir_sm)
print('')
os.chdir(origdir)
print('os.getcwd():',os.getcwd())
print('')
tm=os.listdir(dir_sm)
#print('**** tm:',tm)
#if os.path.isdir('eval_performance'):
if 'eval_performance' in tm:
    tm.remove('eval_performance')
if 'storage' in tm:	
    tm.remove('storage')
if 'png' in tm:	
    tm.remove('png')
    
print('trained models:')
print(tm)
print('')

clfs=list(set([ iqq.split('.')[-3] for iqq in tm ]))

os.chdir(inference_dir)
dir_app=os.getcwd()
print('dir_app:',dir_app)

### abstract *.npy files from all class folders in inference_dir/
#os.system('cp ../ln_ta_fr_cls_dirs.sh .; ./ln_ta_fr_cls_dirs.sh; rm -rf cp_ta_fr_cls_dirs.sh')
print('')

countclass={}
klas=list(set([ iqq.split('.')[-2] for iqq in tm ]))

### defining others
others=''
for j in votingclass:
    if not j in klas:
        others=j

cutoff=int(len(klas)*len(clfs)*0.55)  #### if score is less than this, its not a winner

print('others, if exist (class to be inferred but never been trained in the models)',others)
print('cutoff, minimal accumulated score by the clf pool for a consentual candidate to be valid',cutoff)

for j in klas:
    countclass[j]=0

npyfiles=[]
for i in os.listdir(dir_app):
    if i.split('.')[-1]=='npy':
        npyfiles.append(i)

print("npyfiles:")
print(npyfiles)
print('')

for i in npyfiles:
    for j in klas:
        if j in i:
            countclass[j]=countclass[j]+1
print('countclass:',countclass)
print('')


# initialise correct
correct={}
for i in klas:
    correct[i]=0    
# end of initialise dict_1

    
totalvote=len(tm)
print('Detailed printout of the inferences made by the trained models:')
print(" ")

# initialise dict_22,dict_33
dict_22={};dict_33={}
for j in tm:
    dict_22[j]=0  
    dict_33[j]=0  
# end of initialise dict_22,dict_33


# initialise clfperf
clfperf={};wclfperf={};
for j in clfs:
    clfperf[j]=0
    wclfperf[j]=0
        
# end initialise clfperf

wor={};countcan=0

for i in npyfiles:
    print('')
    print('********************************** New fingerprint **********************************')
    print('')
#### initialise sumverclfs[cls] dictionary for the specific i ###
    accmclassiq={}
    for dumm in klas:
        accmclassiq[dumm]=0
#### end of initialise sumverclfs[cls] dictionary for the specific i ###
    # initialise dict_1
    dict_1={}
    for ii in ['is', 'is_not']:
        for jj in tm:
            clfclsq=jj.split('.')[-2]
                #print(i,targetclass,j)
            isornotq=ii
            dict_1[isornotq,clfclsq]=0
    # end of initialise dict_1
	
    delta={}
    for j in tm:
        targetclass=i.split('.')[0]
        clfcls=j.split('.')[-2]
        jclf=j.split('.')[1]
        
        ##### do inference here ################
        #print(i,targetclass,j)
        modelfile2load=os.path.join(dir_sm,j)
        #print('targetclass,modelfile2load:',targetclass,modelfile2load,os.path.isfile(modelfile2load))
        #print('i:',i,os.path.isfile(i))
        
        with open(modelfile2load, 'rb') as f:
            trained_classifier = pickle.load(f)
        #trained_classifier=pickle.load(trained_classifier)
        X_test=np.load(i)
        X_test=X_test.reshape(1,X_test.shape[0]*X_test.shape[0])
        y_pred = trained_classifier.predict(X_test)[0]
        #print('y_pred:',y_pred)
        if y_pred == 0:
            isornot='is'
        elif y_pred == 1:
            isornot='is_not'
        ##### end of do inference here ################      
        #print('y_pred:',y_pred,';',j,'says',i,isornot,clfcls)
        
        ###### count if clf infers correctly
        isplit=i.split('.')[0]
        if isplit==clfcls and isornot=='is':
            clfperf[jclf]=clfperf[jclf]+1
#            print('y_pred:',y_pred,';',j,'infers correctly',i,isornot,clfcls,jclf,clfperf[jclf])
            dict_22[j]=dict_22[j]+1
            delta[jclf,clfcls]=1-int(y_pred)
            print('y_pred:',y_pred,"delta("+jclf+","+clfcls+"):",delta[jclf,clfcls],j,'infers correctly',i,isornot,clfcls,"clfperf["+jclf+"]:",clfperf[jclf])
#            print("delta("+jclf+","+clfcls+"):",delta[jclf,clfcls])
            
        elif isplit==clfcls and isornot=='is_not':
            wclfperf[jclf]=wclfperf[jclf]+1
            dict_33[j]=dict_33[j]+1
            delta[jclf,clfcls]=1-int(y_pred)
            print('y_pred:',y_pred,"delta("+jclf+","+clfcls+"):",delta[jclf,clfcls],j,'infers INcorrectly',i,isornot,clfcls,"wclfperf["+jclf+"]:",wclfperf[jclf])
#            print("delta("+jclf+","+clfcls+"):",delta[jclf,clfcls])
            
        elif isplit!=clfcls and isornot=='is_not':
            clfperf[jclf]=clfperf[jclf]+1
            dict_22[j]=dict_22[j]+1
            delta[jclf,clfcls]=1-int(y_pred)
            print('y_pred:',y_pred,"delta("+jclf+","+clfcls+"):",delta[jclf,clfcls],j,'infers correctly',i,isornot,clfcls,"clfperf["+jclf+"]:",clfperf[jclf])
#            print("delta("+jclf+","+clfcls+"):",delta[jclf,clfcls])
            
        elif isplit!=clfcls and isornot=='is':
            wclfperf[jclf]=wclfperf[jclf]+1
            dict_33[j]=dict_33[j]+1
            delta[jclf,clfcls]=1-int(y_pred)
            print('y_pred:',y_pred,"delta("+jclf+","+clfcls+"):",delta[jclf,clfcls],j,'infers INcorrectly',i,isornot,clfcls,"wclfperf["+jclf+"]:",wclfperf[jclf])
#            print("delta("+jclf+","+clfcls+"):",delta[jclf,clfcls])
 
        dict_1[isornot,clfcls]=dict_1[isornot,clfcls]+1
    #print('%%% dict_1:',dict_1)
    for isornot in ['is', 'is_not']:    
        for j1 in klas:
            dict_1[isornot,j1]=dict_1[isornot,j1]/totalvote
    #print(' ')
    s1=sorted(dict_1.items(), key=lambda x:x[1])
    s1.reverse()
    
    ## work here 
    print('')
    for j2clf in clfs:
       
        for iq in range(len(klas)):
            classiq=klas[iq]
            dclasses=klas[0:iq] + klas[iq+1:]
            nscoreI=delta[j2clf,classiq]
            nscoreIp=[ 1-delta[j2clf,dum] for dum in dclasses ]
                        
            if nscoreI==0 and np.sum(nscoreIp) == len(dclasses):
                print('overwrite nscore because', i,'is not in',klas)
                nscoreI=0
                nscoreIp=[ 0 for dum in dclasses ]
                
            nscore=nscoreI+np.sum(nscoreIp)                
            accmclassiq[classiq]=accmclassiq[classiq]+nscore
            print('classiq:',classiq,'nscoreI:',nscoreI,'dclasses:',dclasses,'nscoreIp:',nscoreIp,'score of item',i,'being',classiq,'by',j2clf,'is',nscore,'accmclassiq['+classiq+']:',accmclassiq[classiq])			
            #print('score of item',i,'being',classiq,'by',j2clf,'is',nscore,'accmclassiq['+classiq+']:',accmclassiq[classiq])
            #print('classiq:',classiq,'j2clf:',j2clf,'accmclassiq['+classiq+']:',accmclassiq[classiq])    
            #print('accmclassiq['+classiq+']:',accmclassiq[classiq])    
        print('')
    
    listdict=[ (classiq,accmclassiq[classiq]) for classiq in klas ]
    listt=[ (listdict[i][0],listdict[i][1]) for i in range(len(listdict))]
    #print('before sorted, listt:',listt)
    listt.sort(key = lambda x: x[1])
    nc=np.sum([listt[i][1] for i in range(len(listt))])
	
    def truncate(number: float, digits: int) -> float:
        pow10 = 10 ** digits
        return number * pow10 // 1 / pow10
	
    display=[ (listt[i][0],listt[i][1],truncate(listt[i][1]/nc,2)) for i in range(len(listt)) ]
    print('Candidates proposed by the collective vote on',i,'listed in increasing relative possibility are')
    print(display)
    sortedscore=[ listt[i][1] for i in range(len(listt)) ]
    
    if sortedscore[-1]!=sortedscore[-2] and sortedscore[-1] >= cutoff:
        winner=listt[-1]
	
    if sortedscore[-1]==sortedscore[-2]:
        winner='undetermined'
		
    if len(set(sortedscore))==1 or sortedscore[-1] < cutoff:
        winner='none of the above'
        
    print('Based on the given probability list, the consensual candidate on',i,'is choosen to be:',winner)
    #print("winner[0],i.split('.')[0]:",winner[0],i.split('.')[0])
    countcan=countcan+1
    if winner[0]==i.split('.')[0]:
        print('Comment for accuracy assessment of the cvp package:',i,'is inferred correctly as',winner[0],'with a highest relative probability',display[-1][2],'=)')
        wor[countcan]=("bingo",i)
    elif (not i.split('.')[0] in klas) and winner=='none of the above':
        print("Comment for accuracy assessment of the cvp package:",i,'is inferred correctly',winner,'=)')
        wor[countcan]=("bingo",i)
    else:
        print("Comment for accuracy assessment of the cvp package:",i,'is inferred INcorrectly','=(')
        wor[countcan]=("flop",i)
    print('  ')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX End of New fingerprint XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print('')
    ## end work here 
 
    news1=[];
    #for qq in s1:
    #    if qq[0][0] == 'is':
    #        jj=qq
    #        news1.append(jj)
    
    #statement=[ i+" "+news1[j][0][0]+' '+news1[j][0][1]+" "+'with prob '+str(news1[j][1]) for j in range(len(news1))]
    #print(' ')
    #print('Sorted results of votes by tm on',i,':')
    #print('------------------------------------------')
    #for i2 in statement:
    #    print(i2)
    
    #if news1[0][0][1] == i.split('.')[0]:
    #    print(news1[0][0][1] ,'is inferred correctly as ', i.split('.')[0])
    #    correct[i.split('.')[0]]=correct[i.split('.')[0]]+1
    #else:
    #    print(i.split('.')[0],'is not inferred correctly' )
    #print(' ')

print("")
print("================")
print("Overall analysis")
print("================")
print('')

print('"others": class, if exist (class to be inferred but never been trained in the models):',others)
print('')
print('cutoff, minimal accumulated score by the clf pool for a consentual candidate to be valid:',cutoff)
print('')
print('maximal accumulated score of a candidate by the clfs voting pool = len(clfs)xlen(klas):',len(clfs)*len(klas))
print('')

wordum=[ (i,wor[i][1],wor[i][0]) for i in wor.keys() ]
print("Breakdown of the fingerprints correctly (bingo) or INcorrecly inferred (flop) by the cvp package")
print('------------------------------------------------------------------------------------------------------------')
for i in wordum:
    print(i[0],i[1],i[2])
print('Total number of fingerprints presented for inference:',len(wor))	
print("")
print('Classifier pool (' + str(len(clfs)) + ' classifiers):')
print(clfs)
print('')
#print('The number of voting classifiers:',len(clfs))
#print('The classes of the training fingerprints (' + str(len(klas)) + ' classes):')
print('Classes for which the classifier pool are trained to recognize (' + str(len(klas)) + ' classes):')
print(klas)
#print('The number of classes of the training fingerprints:',len(klas))
print('')
print('Classes of fingerprints to be inferred ('+ str(len(votingclass)) + ' classes):',votingclass)
print('The "others" class in the classes of the fingerprints to be inferred (if exist):',others)
#print('The number of classes of the inference fingerprints:',len(votingclass))
countbingo=[ wor[i+1][0] for i in range(len(wor)) ].count('bingo')
countflop=[ wor[i+1][0] for i in range(len(wor)) ].count('flop')
voteperformance=str(100*countbingo/len(wor))+"%"

print('')
print('The performance of each classifier in inferencing these',len(wor),'fingerprints. Note that each classifer has to provide',len(klas),'inferences of either "yes" or "no" to each fingerprint in the inference_dir/ directory')
lclfperf=list(set(clfperf))
lwclfperf=list(set(wclfperf))
totalattempt=[ (wclfperf[i]) for i in clfperf ][0]+[ (clfperf[i]) for i in clfperf ][0]

clfperflist=[ (i,100*clfperf[i]/totalattempt) for i in clfperf ]
clfperflist.sort(key = lambda x: x[1])
clfperflist.reverse()

print("classifier  count of correct inference 	count of INcorrect inference  total attempts  accuracy(in %)")
print("----------  --------------------------	----------------------------  --------------  --------------")
for i in range(len(clfperflist)): 
    print(clfperflist[i][0],clfperf[clfperflist[i][0]],wclfperf[clfperflist[i][0]],totalattempt,clfperflist[i][1])
#print('')
#print('Accuracy performance (in %) of the classifiers in correctly recoginising the fingerprints:')
#print(clfperflist)
print('')


print("inference of fingerprints excluding the 'others' class")
td=0
print("class", "Number of sample inferred", "count of correct inference", "accuracy of correct inference(%)")
print("-----", "-------------------------", "--------------------------", "--------------------------------")
sumnumberofi=0
for i in klas:
#for i in votingclass:
    numberofi=[ wor[i+1][1].split(".")[0] for i in range(len(wor)) ].count(i)
    countbingoi=[ (wor[i+1][1].split(".")[0],wor[i+1][0]) for i in range(len(wor)) ].count((i,'bingo'))
    try:
        percentage=100*(countbingoi/numberofi)
    except:
        percentage='nan'
    print(i,numberofi, countbingoi,percentage)
#    sumnumberofi=sumnumberofi+numberofi
    td=td+countbingoi; sumnumberofi=sumnumberofi+numberofi
    
print('Total number of fingerprints inferred (excluding the "others" class):',sumnumberofi)
print('Total number of fingerprints correctly inferred (excluding the "others" class):',td)
print('Accuracy (excluding the "others" class):',truncate(100*td/sumnumberofi,1),'%')


if not others =='':
    print('')
    print('Here, the "others class is"',others)
    accum=0;aothers=0
    print("There are fingerprints of 'others' class to be inferred")
    print("The classifiers in the voting pool are not trained to recognize them")
    print("These fingerptins, if inferred correctly, should be classified as 'none of the above'")
    print("The following are the results of the inference of the 'others' fingerprints")
    print('----------------------------------------------------------------------------')
    for i in wordum:
        if i[1].split('.')[0]==others and i[2]=='bingo':
            accum=accum+1    
        if i[1].split('.')[0]==others:
#            print(i[0],i[1],i[2])
            aothers=aothers+1
    print('')            
    print("'bingo' means the fingerprint is correctly catagorized as 'none of the above'")
    print("'flop' means the fingerprint is not correctly catagorized as 'none of the above'")
    print(accum,'out of',aothers,'fingerprints of class "others" in voting/ is correctly inferred. Percentage of this accuracy is',truncate(100*accum/(aothers),1),'%')

nothers=len(wor)-sumnumberofi
print('Number of "others" class in the inference fingerprints presented:',nothers)
print('')

try:
    totalbingo=accum+td
except:
    totalbingo=td	

print('Total number of fingerprints correctly inferred by the cvp (inclusive of the "others" fingerprints) is',totalbingo,'out of',len(wor),"("+voteperformance+")")

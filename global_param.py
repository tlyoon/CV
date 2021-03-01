#Nnofs                              #### multiplication factor for oversampling training data set
                                    #### Final number of samples in a class is Nnofs*NALL; where NALL is the sum                                      #### of fingerprints in all classes
Nnofs = 5                           #### For production, use = 5; For trouble shooting, set to 1

#Nnofs_evaluate                     #### multiplication factor for oversampling evaluation data set
Nnofs_evaluate = 1                  #### default = 20; for trouble shooting, set to 2
				    #### irrelevant if run main_vote.py

#Nround_evaluate                    #### No. of rounds to perform evaluation
Nround_evaluate = 1                 #### default = 20; for trouble shooting, set to 2
				    #### Set to 1 if run main_vote.py

#fractrain                          #### 0.50 the default ratio for splitting the original data in the class/sample subdirectories
                                    #### of all classes into train and authetic samples according to 
                                    #### Say if a class/ has 10 ta.npy files in its subdirectory, 
                                    #### int(fractrain*10) of them will be used as 'seed' for generating
                                    #### over-sampling (train) data for training purpose, the rest, 10 - int(fractrain*10) will
                                    #### remain as 'authentic' data.                              
fractrain = 0.0                     ##### default = 0.5. Do not change this unnecessarily
                                    #### If set to = 0.0, only data for trainning are generated. No data for evaluation are generated.
                                    #### Set to = 0.0 if run main_vote.py


#nephochs                           #### parameter required by keras.Sequential
nephochs=10 

#sampler_train = 'SMOTE'            #### imbalanced-learn sampler to be deployed for oversampling training data. Default is 'SMOTE'
#sampler_train = 'BorderlineSMOTE'
#sampler_train = 'ADASYN'
sampler_train = 'RandomOverSampler'
#sampler_train = 'SVMSMOTE'
#sampler_train = 'KMeansSMOTE'       #### 'KMeansSMOTE'does not seem to work


#sampler_eval  = 'SMOTE'             #### imbalanced-learn sampler to be deployed for oversampling evaluation data. Default is 'SMOTE'
#sampler_eval = 'ADASYN'
sampler_eval = 'RandomOverSampler'
#sampler_eval = 'SVMSMOTE'
#sampler_eval = 'KMeansSMOTE'       #### 'KMeansSMOTE'does not seem to work

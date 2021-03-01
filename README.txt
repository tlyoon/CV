"Collective Voting" Classification Scheme for 2D FTIR herbal fingerprints
=========================================================================

The file structure of the CV package is as follows:

main/ --
	   |-- *.py, *.sh
	   |--dataset1/
	   |--dataset2/
	   
2. The directories dataset1/ and dataset2/ contains sample raw data of 5 classes of herbal type. Download dataset1/ and dataset2/ seperately from https://staffusm-my.sharepoint.com/:f:/g/personal/tlyoon_usm_my/Ej3EcxTvnZBLt2rqW-HhboUBoGoEjd2WRaCihvfDveRWcw?e=iDrEh2 and https://staffusm-my.sharepoint.com/:f:/g/personal/tlyoon_usm_my/Ej3EcxTvnZBLt2rqW-HhboUBoGoEjd2WRaCihvfDveRWcw?e=iDrEh2.

3. The package is devided into two independent parts: (i) Generation of 2D FTIR fingerprints and (ii) Implementation of the CV package using the fingerprints.

(i) Generation of 2D FTIR fingerprints
    ==================================
The dataset1/ and dataset2/ directories contain samples of raw IR data files with *.asc suffix. Use gen_fgprint_v2.py in conjunction with gpu_v2.py to integrate the *.asc files into two data files, namely ta.npy and ts.npy (and its corresponding *.png version) in the directories of all herbal samples. To do so, organise the data structure as depicted below:

 dataset/ 
		|--gen_fgprint_v2.py, gpu_v2.py
		|--JXL/
			|--JXL-J01/
					|--JXL-J01-20C.asc
					|--JXL-J01-30C.asc
					|--JXL-J01-40C.asc
					|--..
					|--JXL-J01-120C.asc
			|--JXL-J02/ 
					|-- ...
					|-- ..
			|--JXL-J03/
					|..
			|--...
		|--MT/
			|--MT-M01/
					|-- ...
					|-- ...
					|-- ...
			|--MT-M02/
					|-- ...
					|-- ...
					|-- ...
		|--SY/
			|-- ..
			|-- ...
		|--GW/
			|-- ..
			|-- ...
		|--ZX/
			|-- ..
			|-- ...

Execute 
	ptyhon gen_fgprint_v2.py

The code check_tsta_records.py can be used to do monitor the real-time progress of the generation of ta.npy (for asynchronous correlation) and ts.npy (for synchronous correlation) files in each of the folders. ta.png and ts.png are the corresponding 2D fingperprints for visualisation. Once all ta.npy and ts.npy files have been correctly prepared, proceed to the implementation step of the CV package using these fingerprints.


(ii) Implementation of the CV package using the fingerprints
     =======================================================
	 
The implementation of the CV package must follow the hieracy of the following data structure:

main/ --
	   |-- *.py, *.sh
	   |--training/
				|--JXL/
					|--JXL-J01/
							|--ta.npy
					|--JXL-J02/ 
							|--ta.npy
					|--JXL-J03/
							|--ta.npy
					|--...
				|--MT/
					|--MT-M01/
							|--ta.npy
					|--MT-M02/
							|--ta.npy
						|-- ...
				|--SY/
					|-- ..
					|-- ...
				|--GW/
					|-- ..
					|-- ...
				|--ZX/
					|-- ..
					|-- ...
		|--inferencing/
				|--JXL/
					|--JXL-J11/
							|--ta.npy
					|--JXL-J12/ 
							|--ta.npy
					|--JXL-J13/
							|--ta.npy
					|--...
				|--MT/
					|--MT-M11/
							|--ta.npy
					|--MT-M12/
							|--ta.npy
						|-- ...
				|--SY/
					|-- ..
					|-- ...
				|--GW/
					|-- ..
					|-- ...
				|--ZX/
					|-- ..
					|-- ...
			

1. Prepare the data in training/ and inference/ manually. We shall demonstrate how this can be done using the *.npy data in /main/dataset1/ or main/dataset2/. For example,

cd main/
mkdir training/
cd training/
ln -s ../dataset2/* .

[tlyoon@anicca training]$ ls
GW/  JXL/  MT/  SY/  ZX/

[tlyoon@anicca training]$ ls GW/
G001/  G002/  G003/  G004/  G005/ 

3. Edit global_param.py global for the value of Nnofs, e.g, for production run, set e.g.,

Nnofs = 5

This is the  multiplication factor for oversampling training data set. Final number of samples in a class is Nnofs*NALL; where NALL is the sum of fingerprints in all classes.

The data in training/ will be later oversampled by data_preprocess_vote_v2.py in main.vote.py. 

4. Prepare the data in /main/inference/ manually.

5. To this end you may use the *.npy data in /main/dataset1/ (or dataset2/ if you have already used dataset1 for training/ in step 2). For example,

cd main/
mkdir inference/
cd inference/
ln -s  ../dataset1/* .

[tlyoon@anicca inference]$ ls
GW  JXL  MT  SY  ZX

ls GW/
G010/  G011/  G012/  G013/  G014/

6. Edit the variable Nnofs in the file oversample.py, e.g., Nnofs=5. The number of oversamples in each class is defined as nofs=Nnofs*maxldf. Note that the number of original samples in each class in general is different. maxldf refers to the number of samples in the largest class.

7. cp ../oversample.py .
   python oversample.py

The *.npy files in the directories, e.g., GW/, JXL/, ..., will be oversampled accordingly and appear in the current directory, 

[tlyoon@anicca inference]$ ls
GW.0.npy   GW.34.npy  JXL.0.npy   JXL.34.npy  MT.0.npy   MT.34.npy  SY.0.npy   SY.34.npy  ZX.0.npy   ZX.34.npy
GW.10.npy  GW.35.npy  JXL.10.npy  JXL.35.npy  MT.10.npy  MT.35.npy  SY.10.npy  SY.35.npy  ZX.10.npy  ZX.35.npy
GW.11.npy  GW.36.npy  JXL.11.npy  JXL.36.npy  MT.11.npy  MT.36.npy  SY.11.npy  SY.36.npy  ZX.11.npy  ZX.36.npy
GW.12.npy  GW.37.npy  JXL.12.npy  JXL.37.npy  MT.12.npy  MT.37.npy  SY.12.npy  SY.37.npy  ZX.12.npy  ZX.37.npy

Note that the original *.npy data in inference/ must not overlap with that in training/ in step (2) above.


11. Once the data in training/ and inference/ has been prepared, 

nohup python main_vote.py &

12. Detailed procedure of what files are run can be inferred easily from main_vote.py. It includes running the folowing packages in tendem: 

ln_train_data.py --> data_preprocess_vote_v2.py  -> training_1vA.py -> mvpng.py -> cvp.py


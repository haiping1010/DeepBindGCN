# DeepBindGCN
## background
The core of large-scale drug virtual screening is to accurately and efficiently select the binders with high affinity from large libraries of small molecules in which non-binders are usually dominant. The protein pocket, ligand spatial information, and residue types/atom types play a pivotal role in binding affinity. Many docking and complexes dependent models have developed to solve this problem, however, most of them are too time and resources comsuming for large scale virtual screening over billions of compounds for given target. The accurate is also questionable becuase most of those existing method seriously lacking non-binding protein-ligand pairs in training set. Here we used the pocket residues or ligand atoms as nodes and constructed edges with the neighboring information to comprehensively represent the protein pocket or ligand in-formation. Moreover, the model with pre-trained molecular vectors performs better than the onehot representation. The main advantage of DeepBindGCN is that it is non-dependent on docking conformation and concisely keeps the spatial information and physical-chemical feature. Notably, the DeepBindGCN_BC has acceptable AUC in many DUD.E datasets, and DeepBindGCN_RG achieves a very low RMSE value in most DUD.E datasets. 

## intallation and dependence
Before use, please installing the environment:

Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric

Install rdkit: conda install -y -c conda-forge rdkit

Or run the following commands to install both pytorch_geometric and rdkit:

conda create -n DeepBindGCN python=3

conda activate DeepBindGCN

conda install -y -c conda-forge rdkit

conda install pytorch torchvision cudatoolkit -c pytorch

pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html

pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html

pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html

pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html

pip install torch-geometric

## Usage of the DeepBindGCN_BC
For using the DeepBindGCN_BC, please check the DeepBindGCN_BC_example's readme.txt file
Before runing the file, please download the trained model from Repository's release file full_model_out2000_BC.model, and put it in the DeepBindGCN_BC_example folder.
----
### Detailed runing DeepBindGCN_BC example is:
initial the env  by typing:
conda activate DeepBindGCN

1. prepare ligands in all_data folder, convert ligand into dictory

cd all_data
bash run_all_dic.bash
cd ..

2. prepare the pocket in the pocket folder

cd pocket
python  extract_pocket.py
cd ..

3. preparing the input dataframe

bash run_all_n_add.bash

4. run the prediction
bash run_all_predict_add.bash

5. sort the result
bash score_sort_add.bash
-----
## Usage of the DeepBindGCN_RG
For using the DeepBindGCN_RG, please check the DeepBindGCN_RG_example's readme.txt file
Before runing the file, please download the trained model from Repository's release file full_model_out2000_RG.model, and put it in the DeepBindGCN_RG_example folder.
---
### Detailed runing DeepBindGCN_RG example is:
initial the env  by typing:
conda activate DeepBindGCN

1. prepare ligands in all_data folder, convert ligand into dictory

cd all_data
bash run_all_dic.bash
cd ..

2. prepare the pocket in the pocket folder

cd pocket
python  extract_pocket.py
cd ..

3. preparing the input dataframe

bash run_all_n_add.bash

4. run the prediction
bash run_all_predict_add.bash

5. sort the result
bash score_sort8.6_BA.bash
----
### Any questions are welcomed to contact the author by email hp.zhang@siat.ac.cn


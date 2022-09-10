# DeepBindGCN

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


For using the DeepBindGCN_BC, please check the DeepBindGCN_BC_example's readme.txt file
Before runing the file, please download the trained model from Repository's release file full_model_out2000_BC.model, and put it in the DeepBindGCN_BC_example folder.

For using the DeepBindGCN_BC, please check the DeepBindGCN_RG_example's readme.txt file
Before runing the file, please download the trained model from Repository's release file full_model_out2000_RG.model, and put it in the DeepBindGCN_RG_example folder.



Any questions are welcomed to contact the author by email hp.zhang@siat.ac.cn



initial the env  by typing:
conda activate DeepBindGCN

1, prepare ligands in all_data folder, convert ligand into dictory

cd all_data
bash run_all_dic.bash
cd ..

2, prepare the pocket in the pocket folder

cd pocket
python  extract_pocket.py
cd ..

3, preparing the input dataframe

bash run_all_n_add.bash

4. run the prediction
bash run_all_predict_add.bash

5. sort the result
bash score_sort8.6_BA.bash


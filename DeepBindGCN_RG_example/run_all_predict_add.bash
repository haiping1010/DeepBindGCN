for name in  data1/processed/L_P_train_BA.pt
do

base=${name:16 }
base_n=${base%.pt}
echo $base
python  training_nn3_load_name.py  $base_n > $base_n'_predict.log'

done




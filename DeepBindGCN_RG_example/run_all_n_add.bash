
name='all_file'
for code in   BA 
#for code in  JD JE JF JG JH JI JJ

do
base=$code
#mkdir -p 'mp_data_'$base
#head -n1 all_file/BAAA_n.csv > $name'/XXXXX_n.txt'

##python  read_smi_protein_nnn_usage.py  $name  'mp_data_'$base

nohup python  read_smi_protein_nnn_usage.py  $name  'mp_data_'$base  > $base'.log' 2>&1&

sleep 1s
echo $base

done

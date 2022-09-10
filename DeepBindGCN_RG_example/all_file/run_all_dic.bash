
for name in ??_n.smi
do
base=${name%.smi}

nohup python  read_smi_protein_nnn.py  $base  > $base_log.log 2>&1&

sleep 4s


done


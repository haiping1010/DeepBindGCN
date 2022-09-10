

sort -g -rk 1,1  output_BA_n.txt  > all_out_BA.sort

awk -F ',' ' $1 >= 8.6 ' all_out_BA.sort  >  all_out_select_BA8.6.sort



cat  output_??_n.txt  > all_out.list

sort -g -rk 1,1 output_BA_n.txt  > all_out_BA.sort

awk -F ',' ' $1 >= 0.99 ' all_out_BA.sort  >  all_out_select_BA.sort


#!bin
nvprof ./Prog 100   &> ./outputmemcpy/out_100.txt
for i in   30000
do
	echo $i 
	nvprof ./Prog $i  &> ./outputmemcpy/out_$i.txt   
done 

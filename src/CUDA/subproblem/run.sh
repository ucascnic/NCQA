#!/bin/bash

echo Making sure program is installed
make install

getopts ":h" opt;
	if [ "$opt" == "h" ] ; then
		echo "Running on head node"
			echo
			echo Starting program in serial
				time ./gen/program s
				echo
			echo Starting program in parallel
				time ./gen/program p
				echo 
	else
		echo Running on cluster option
			echo
			echo Starting program in serial
				time ./gen/program s
				echo
			echo Starting program in parallel
				time ./gen/program p			
				echo
	fi
echo Done


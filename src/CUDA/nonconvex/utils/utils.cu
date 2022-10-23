#include <utils/utils.h>

#include <fstream>

namespace utils{
static std::string LOG_FILE = "log.txt";

void LogProgramResults(std::string name, HostD vector){
	LOG_FILE = "output.txt";
	Log("===========================================================================================================\n\n                                  " + name + "                                         \n\n===========================================================================================================");
	
	PrintVector(true,"X",vector);
	LOG_FILE = "log.txt";
}


void PrintVector(std::string name, HostD vector){
	for(int i=0; i <vector.size(); i++){
		std::cout << name << "[" << i << "]= " << vector[i] << std::endl;
	}
	std::cout << std::endl;
}


void PrintVector(std::string name, DeviceD d_vector){
	HostD h_vector = d_vector;
	PrintVector(name, h_vector);

}


void PrintVector(bool log, std::string name, HostD vector){
	if(log){
		std::ofstream out(LOG_FILE,std::ios::out | std::ios::app);
		if(out.is_open()){
			std::streambuf *old_buffer = std::cout.rdbuf();
			std::cout.rdbuf(out.rdbuf());
		
			PrintVector(name,vector);
	
			std::cout.rdbuf(old_buffer);
			out.close();
		} else{
			std::cout << "Could not open log file" << std::endl;
		}
	} else{
		PrintVector(name, vector);
	}
}

void Log(std::string text){

	std::ofstream out(LOG_FILE, std::ios::out | std::ios::app);
	if(out.is_open()){
		out << text << "\n";
		out.close();
	} else{
		std::cout << "Could not open log file" << std::endl;
	}

}
}//END - namespace


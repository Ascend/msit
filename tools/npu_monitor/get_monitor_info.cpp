#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "dcmi_interface_api.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

#define MAX_CARD_NUM (16)
#define NPU_OK (0)

int main(int argc, char ** argv)
{
    int ret;
    int card_count = 0;
    int card_id_list[MAX_CARD_NUM] = {0};

    ret = dcmi_init();
    if (ret != NPU_OK) {
        printf("Failed to init dcmi.\n");
        return ret;
    }

    ret = dcmi_get_card_num_list(&card_count, card_id_list, MAX_CARD_NUM);
    if (ret != NPU_OK) {
        printf("Failed to get card number.\n");
        return ret;
    }
    //printf("card count is %d\n", card_count);
    //获取card ID
    int card_num = 0;
    int card_list[16] = {0};
    ret = dcmi_get_card_list(&card_num, card_list, 16);
    
    for(int i = 0; i < 1; ++i){
    	int ret = 0;
    	std::cout << std::left << std::setw(5) << "ID" << std::setw(10) << "  AI Core(%)  " << std::setw(10)  << "HBM(MB)" << std::setw(10)  << "Power(W)"  << std::endl;
    	std::cout << "--------------------------------------" << std::endl;

    	for (int id = 0; id < card_count; ++id) {
		int card_id = card_id_list[id];
		int device_id = 0;

		int input_type = 2;
		unsigned int utilization_rate = 0;
		int ret_utilization = dcmi_get_device_utilization_rate(card_id, device_id, input_type, &utilization_rate);
        	//struct dcmi_memory_info_stru pdevice_memory_info = {0};
		//int ret_hbm = dcmi_get_memory_info(card_id, device_id, &pdevice_memory_info);

		struct dcmi_hbm_info device_hbm_info = {0};
		int ret_hbm = dcmi_get_device_hbm_info(card_id, device_id, &device_hbm_info);
	
		int power = 0;
		int ret_power = dcmi_get_device_power_info(card_id, device_id, &power);
		
		//std::cout << ret_utilization << std::endl;
		//std::cout << ret_hbm << std::endl;
		//std::cout << ret_power << std::endl;	
		if (ret_utilization == 0 && ret_hbm == 0 && ret_power == 0) {
            		std::cout << std::setw(10) << card_id << std::setw(10) << utilization_rate << std::setw(10) << device_hbm_info.memory_usage << std::setw(10) << std::fixed << std::setprecision(1) << power / 10.0  << std::endl;
        	} else if(ret_utilization!=0) {
            		std::cerr << "Error: Unable to get utilization rate for card " << card_id << std::endl;
        	}else if (ret_hbm!=0){
	    		std::cerr << "Error: Unable to get hbm info for card " << card_id << std::endl;
		}else{
	    		std::cerr << "Error: Unable to get power info for card " << card_id << std::endl;
		}

		string dir_path = "record";
		mkdir(dir_path.data(),0755);

		std::ofstream file(dir_path + "/device_" + std::to_string(card_id) + ".csv", std::ios::app);
		if (file.is_open()) {
			if (file.tellp() == 0) {
				file << "utilization,HBM,power\n";
			}
        
        		file << utilization_rate << ","
             		     << device_hbm_info.memory_usage << ","
             		     << std::fixed << std::setprecision(1) << power/10.0 << "\n";
        		file.close();
    		} else {
        		std::cerr << "Error: Unable to open file for device " << device_id << std::endl;
    		}
    	}
    }
    return ret;
}

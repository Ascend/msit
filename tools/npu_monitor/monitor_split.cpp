#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

void writeCSV(int file_count,int device_id, const vector<vector<string>>& data) {
    /*filesystem::path dir_path = "dataset"+tostring(file_count);
    if (!filesystem::exists(dir_path)) { 
    	filesystem::create_directory(dir_path);
    }*/
    string dir_path = "task"+to_string(file_count);
    mkdir(dir_path.data(),0755);
    std::cerr << "device_"+to_string(device_id)+".csv" << std::endl;
    string filename = "device_"+to_string(device_id)+".csv";
    ofstream file(dir_path+"/"+filename);
    file << "utilization,HBM,power\n";
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << endl;
    }

    file.close();
}

// 函数：处理 CSV 数据并根据条件分割文件
void processCSV(int device_id) {
    string filename = "record/device_"+to_string(device_id)+".csv";
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
    }

    std::string line;
    vector<vector<string>> zero_data;
    vector<vector<string>> current_data;
    int zero_count = 0;
    int file_count = 1;
    bool start = false;
    bool first_row=true;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string utilization_str, hbm_str, power_str;
	std::getline(ss, utilization_str, ',');
        std::getline(ss, hbm_str, ',');
        std::getline(ss, power_str, ',');
	int utils = std::stoi(utilization_str);
	if (utils==0 && !start){
		continue;
	}
	start = true;
        if (utils == 0) {
		zero_count++;
		zero_data.push_back({utilization_str, hbm_str, power_str});
	}else if (utils!=0 && zero_count > 10) {
	    writeCSV(file_count,device_id, current_data);

        // 重置计数器和当前数据
        zero_count = 0;
        current_data.clear();
	    zero_data.clear();
        file_count++;
	    std::vector<std::string> row = {utilization_str, hbm_str, power_str};
            current_data.push_back(row);
        }else{
		if (zero_count>0){
			for (const auto& row : zero_data) {
				current_data.push_back(row);
			}
			zero_count = 0; 
			zero_data.clear();
		}
        std::vector<std::string> row = {utilization_str, hbm_str, power_str};
        current_data.push_back(row);
	}
	
    }

    // 写入最后的数据
    if (!current_data.empty()) {
        writeCSV(file_count,device_id, current_data);
    }
    cout << "Data has been split into " << file_count << " files." << endl;
}

int main() {
    for (int i=0;i<8;i++){
    	processCSV(i);
    }
    return 0;
}


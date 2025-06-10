#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Need to add <file_path>" << std::endl;
        std::cerr << "./calculate ${file_path}" << std::endl;
        return 1;
    }
    std::string file_path = argv[1];

    std::ifstream file(file_path);
    if (!file.is_open()) {
	std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::string line;
    int count = 0;
    double total_utilization = 0.0;
    double total_hbm = 0.0;
    double total_power = 0.0;
    double max_utilization = 0.0;
    double max_hbm = 0.0;
    double max_power = 0.0;

    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string utilization_str, hbm_str, power_str;

        std::getline(ss, utilization_str, ',');
        std::getline(ss, hbm_str, ',');
        std::getline(ss, power_str, ',');

        double utilization = std::stod(utilization_str);
        double hbm = std::stod(hbm_str);
        double power = std::stod(power_str);

        if (utilization > max_utilization) max_utilization = utilization;
        if (hbm > max_hbm) max_hbm = hbm;
        if (power > max_power) max_power = power;

        total_utilization += utilization;
        total_hbm += hbm;
        total_power += power;

        count++;
    }

    if (count > 0) {
        double avg_utilization = total_utilization / count;
        double avg_hbm = total_hbm / count;
        double avg_power = total_power / count;

        std::cout << "Average utilization: " << avg_utilization << "%" << std::endl;
        std::cout << "Average HBM: " << avg_hbm << "MB" << std::endl;
        std::cout << "Average power: " << avg_power << "W" << std::endl;
        std::cout << "Max utilization: " << max_utilization << "%" << std::endl;
        std::cout << "Max HBM: " << max_hbm << "MB" << std::endl;
        std::cout << "Max power: " << max_power << "W" << std::endl;
    } else {
        std::cout << "No data to calculate averages." << std::endl;
    }

    file.close();
    return 0;
}


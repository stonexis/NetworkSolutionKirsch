#include "gendata.hpp"


using namespace std;

int main() {
    //Параметры датасета
    constexpr std::size_t num_samples_train = 20000;
    constexpr std::size_t num_samples_val = 2000;
    constexpr std::size_t num_samples_test  = 10;

    constexpr std::size_t num_nodes = 128; //разрешение сетки, количество пикселей

    //Пластина от -1 до 1 
    float xmin = -1.0, xmax = 1.0;
    float ymin = -1.0, ymax = 1.0;

    // Интервалы для a и sigma
    float a_min = 0.05, a_max = 0.1;  
    float sigma_min = 2e3, sigma_max = 1e4;
    std::size_t seed = 42;
    
    auto train_data = generate_data<num_nodes, num_samples_train>(
                                                        seed,
                                                        std::make_pair(xmin,xmax), 
                                                        std::make_pair(ymin,ymax), 
                                                        std::make_pair(sigma_min, sigma_max), 
                                                        std::make_pair(a_min, a_max)
                                                    );
    auto val_data = generate_data<num_nodes, num_samples_val>(
                                                        seed,
                                                        std::make_pair(xmin,xmax), 
                                                        std::make_pair(ymin,ymax), 
                                                        std::make_pair(sigma_min, sigma_max), 
                                                        std::make_pair(a_min, a_max)
                                                    );
    auto test_data = generate_data<num_nodes, num_samples_test>(
                                                        seed,
                                                        std::make_pair(xmin,xmax), 
                                                        std::make_pair(ymin,ymax), 
                                                        std::make_pair(sigma_min, sigma_max), 
                                                        std::make_pair(a_min, a_max)
                                                    );
    
    std::string train_filename_target = "train_target_fields.npy";
    std::string train_filename_input = "train_input_params.npy";
    
    std::string test_filename_target = "test_target_fields.npy";
    std::string test_filename_input = "test_input_params.npy";
    
    std::string val_filename_target = "val_target_fields.npy";
    std::string val_filename_input = "val_input_params.npy";
                                               
    save_tensor_target_nocopy(*train_data.first, train_filename_target);
    save_tensor_input_nocopy(*train_data.second, train_filename_input);
    
    save_tensor_target_nocopy(*test_data.first, test_filename_target);
    save_tensor_input_nocopy(*test_data.second, test_filename_input);

    save_tensor_target_nocopy(*val_data.first, val_filename_target);
    save_tensor_input_nocopy(*val_data.second, val_filename_input);
                                                    
    return 0;
}
#pragma once
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <string>
#include <cmath>
#include <stdexcept>
#include "cnpy.h"

/**
 * @brief Функция генерации равномерной сетки на интервале [a,b]
 * @return array<float, N> Равномерная сетка
 */
template<std::size_t NumNodes>
std::array<float, NumNodes> gen_uniform_grid(float a, float b){
    if ((b - a) < std::numeric_limits<float>::epsilon()) throw std::invalid_argument("Invalid a, b values");
    std::array<float, NumNodes> grid;
    float step = (b - a) / (NumNodes - 1);
    for (std::size_t i = 0; i < NumNodes; i++)
        grid[i] = a + step * i;
    if (grid.back() != b)
        grid.back() = b;
    return grid;
}

/**
 * @brief Функция для заполнения тензора полей напряжений (Таргет), для заданного семпла
 * @return std::unique_ptr<TensorTypeTarget<NumNodes>> - 3-х канальный тензор размерности (3, NumNodes, NumNodes)
 * @note 0-й канал - sigma_xx, 1-й канал - sigma_yy, 2-й канал - sigma_xy
*/
template<std::size_t NumNodes>
std::unique_ptr<TensorTypeTarget<NumNodes>> gen_sigma_field(
                                    const std::array<float, NumNodes>& grid_x,
                                    const std::array<float, NumNodes>& grid_y,
                                    float hole_radius, float P
                                    ){
    if (grid_x.size() != grid_y.size()) throw std::invalid_argument("Invalid grid sizes");                                        
    auto field = std::make_unique<TensorTypeTarget<NumNodes>>();
                                                                                
    for(std::size_t i = 0; i < grid_y.size(); ++i){
        float y = grid_y[i];
        for(std::size_t j = 0; j < grid_y.size(); ++j){
            float x = grid_x[j];
            auto [sigma_xx, sigma_yy, sigma_xy] = UniaxialStress(x, y, hole_radius, P);
            //Карта напряжений (таргет)
            (*field)[0][i][j] = sigma_xx;
            (*field)[1][i][j] = sigma_yy;
            (*field)[2][i][j] = sigma_xy;      
        }
    }
    return field;                                                                            
}
/**
 * @brief Функция для построения тензора параметров и коордиант (Input), для заданного семпла
 * @return std::unique_ptr<TensorTypeInput<NumNodes>> - 4-х канальный тензор размерности (4, NumNodes, NumNodes)
 * @note 0-й канал - значение координаты x в каждой точке сетки
 *       1-й канал - значение координаты x в каждой точке сетки
 *       2-й канал - значение радиуса отверстия в каждой точке сетки (константная карта)
 *       3-й канал - значение нагрузки для пластины P в каждой точке сетки (константная карта) 
*/
template<std::size_t NumNodes>
std::unique_ptr<TensorTypeInput<NumNodes>> gen_mesh_with_params(
                                    const std::array<float, NumNodes>& grid_x, 
                                    const std::array<float, NumNodes>& grid_y,
                                    float hole_radius, float P
                                ){
    if (grid_x.size() != grid_y.size()) throw std::invalid_argument("Invalid grid sizes");
    auto mesh = std::make_unique<TensorTypeInput<NumNodes>>();
    for(std::size_t i = 0; i < grid_y.size(); ++i){
        float y = grid_y[i];
        for(std::size_t j = 0; j < grid_x.size(); ++j){
            float x = grid_x[j];
            //Карта сетки (input)
            (*mesh)[0][i][j] = x;
            (*mesh)[1][i][j] = y;
            //Константные каналы параметров
            (*mesh)[2][i][j] = hole_radius; 
            (*mesh)[3][i][j] = P;
        }
    }
    return mesh;
}

/**
 * @brief Функция для вычисления значений sigma в заданной точке (x,y) для значений радиуса a и нагрузки P
 * @return std::tuple<float, float, float> - (sigma_xx sigma_yy sigma_xy)
*/
std::tuple<float, float, float> UniaxialStress(float x, float y, float a, float P){
    float r = std::sqrt(x*x + y*y);
    float sigma_xx, sigma_yy, sigma_xy;
    if (r < a) {
        // Точка внутри отверстия
        sigma_xx = std::numeric_limits<float>::quiet_NaN();
        sigma_yy = std::numeric_limits<float>::quiet_NaN();
        sigma_xy = std::numeric_limits<float>::quiet_NaN();
    }
    else {
        float theta = std::atan2(y, x);
        float cos2t = std::cos(2.0 * theta);
        float sin2t = std::sin(2.0 * theta);

        float a2 = a * a;
        float a4 = a2 * a2;
        float r2 = r * r;
        float r4 = r2 * r2;

        sigma_xx = (P/2) * (1.0 - (a2 / r2) + (1.0 - (4*a2)/r2 + (3*a4)/r4) * cos2t);

        sigma_yy = (P/2) * (1.0 + (a2 / r2) - (1.0 + (3*a4)/r4) * cos2t);

        sigma_xy = (P/2) * (1.0 - (2*a2)/r2 + ((3*a4)/r4)) * sin2t;
    }
    return std::make_tuple(sigma_xx, sigma_yy, sigma_xy);
}
/**
 * @brief Функция для генерации пары тензоров Target - Input в количестве NumSamples
 * @param seed Зерно генератора случайных чисел
 * @param dispersion_x Интервал границ генерации пара значений - (x_min, x_max) 
 * @param dispersion_y Интервал границ генерации пара значений - (y_min, y_max)
 * @param dispersion_sigma Интервал границ генерации пара значений - (P_min, P_max)
 * @param dispersion_hole_radius Интервал границ генерации пара значений - (a_min, a_max)
 * @return std::pair<
                std::unique_ptr<std::array<TensorTypeTarget<NumNodes>, NumSamples>>,
                std::unique_ptr<std::array<TensorTypeInput<NumNodes>, NumSamples>>
                >
 * @note пара указателей на массивы тензоров Target Input размером Numsamples  
 */
template <std::size_t NumNodes, std::size_t NumSamples>
DataReturnTypeGenData<NumNodes, NumSamples> generate_data(
                                std::size_t seed, 
                                std::pair<float, float> dispersion_x,
                                std::pair<float, float> dispersion_y,
                                std::pair<float, float> dispersion_sigma,
                                std::pair<float, float> dispersion_hole_radius
                                ){
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_a(dispersion_hole_radius.first, dispersion_hole_radius.second);
    std::uniform_real_distribution<float> dist_sigma(dispersion_sigma.first, dispersion_sigma.second);

    auto grid_x = gen_uniform_grid<NumNodes>(dispersion_x.first, dispersion_x.second);
    auto grid_y = gen_uniform_grid<NumNodes>(dispersion_y.first, dispersion_y.second);

    auto Target = std::make_unique<std::array<TensorTypeTarget<NumNodes>, NumSamples>>(); // (N, C_out, NumNodes, NumNodes) -> (NumSamples, 3, NumNodes, NumNodes)
    auto Input = std::make_unique<std::array<TensorTypeInput<NumNodes>, NumSamples>>(); // (N, C_in, NumNodes, NumNodes) -> (NumSamples, 4, NumNodes, NumNodes)                                           

    for(std::size_t s = 0; s < NumSamples; s++){
        float a_val = dist_a(gen);
        float sigma_val = dist_sigma(gen);
        auto field_sample = gen_sigma_field<NumNodes>(grid_x, grid_y, a_val, sigma_val);
        auto params_map = gen_mesh_with_params<NumNodes>(grid_x, grid_y, a_val, sigma_val);
        (*Target)[s] = std::move(*field_sample);
        (*Input)[s] = std::move(*params_map); 
    }
    return {std::move(Target), std::move(Input)};
}

template <std::size_t NumNodes, std::size_t NumSamples>
void save_tensor_target_nocopy(const std::array<TensorTypeTarget<NumNodes>, NumSamples>& data, const std::string& filename){
    constexpr std::size_t elems = NumSamples * 3 * NumNodes * NumNodes;
   
    const float* raw = &data[0][0][0][0]; //указатель на самый первый элемент - скаляр

    // (NumSamples, 3, NumNodes, NumNodes)
    std::vector<std::size_t> shape = { NumSamples, 3, NumNodes, NumNodes };

    cnpy::npy_save(filename, raw, shape, "w"); 
}

template <std::size_t NumNodes, std::size_t NumSamples>
void save_tensor_input_nocopy(const std::array<TensorTypeInput<NumNodes>, NumSamples>& data, const std::string& filename){
    constexpr std::size_t elems = NumSamples * 4 * NumNodes * NumNodes;
   
    const float* raw = &data[0][0][0][0];

    // (NumSamples, 4, NumNodes, NumNodes)
    std::vector<std::size_t> shape = { NumSamples, 4, NumNodes, NumNodes };

    cnpy::npy_save(filename, raw, shape, "w"); 
}
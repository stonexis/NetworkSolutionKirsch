#pragma once
#include <utility>
#include <string>
#include <array>
#include <memory>

template<std::size_t NumNodes>
using Matrix = std::array<std::array<float, NumNodes>, NumNodes>; // 2D N×N matrix

template<std::size_t NumNodes>
using TensorTypeTarget = std::array<Matrix<NumNodes>, 3>;// 3 matrices

template<std::size_t NumNodes>
using TensorTypeInput = std::array<Matrix<NumNodes>, 4>;// 4  matrices

template <std::size_t NumNodes, std::size_t NumSamples>
using DataReturnTypeGenData = std::pair<
    std::unique_ptr<std::array<TensorTypeTarget<NumNodes>, NumSamples>>,
    std::unique_ptr<std::array<TensorTypeInput<NumNodes>, NumSamples>>
>;

template <std::size_t NumNodes, std::size_t NumSamples>
DataReturnTypeGenData<NumNodes, NumSamples> generate_data(
                std::size_t seed, 
                std::pair<float, float> dispersion_x,
                std::pair<float, float> dispersion_y,
                std::pair<float, float> dispersion_sigma,
                std::pair<float, float> dispersion_hole_radius
                ); 

template<std::size_t NumNodes>
std::array<float, NumNodes> gen_uniform_grid(float step, float a, float b);

template<std::size_t NumNodes>
std::unique_ptr<TensorTypeTarget<NumNodes>> gen_sigma_field(
                                                    const std::array<float, NumNodes>& grid_x,
                                                    const std::array<float, NumNodes>& grid_y,
                                                    float hole_radius, float P
                                                    );

template<std::size_t NumNodes>
std::unique_ptr<TensorTypeInput<NumNodes>> gen_mesh_with_params(
                                    const std::array<float, NumNodes>& grid_x, 
                                    const std::array<float, NumNodes>& grid_y,
                                    float hole_radius, float P
                                );

std::tuple<float, float, float> UniaxialStress(float x, float y, float a, float P);

template <std::size_t NumNodes, std::size_t NumSamples>
void save_tensor_target_nocopy(const std::array<TensorTypeTarget<NumNodes>, NumSamples>& data, const std::string& filename);

template <std::size_t NumNodes, std::size_t NumSamples>
void save_tensor_input_nocopy(const std::array<TensorTypeInput<NumNodes>, NumSamples>& data, const std::string& filename);

#include "gendata.tpp"
#pragma once
#ifndef CUDA_CACHE_UTIL
#define CUDA_CACHE_UTIL

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Convenient CUDA error checking macro using existing infrastructure
#define CUDA_CHECK(call) cutilSafeCall(call)

struct CUDACachedFrame {
private:
	bool m_allocated = false;
	unsigned int m_width = 0;
	unsigned int m_height = 0;

	// Helper method to safely free a single pointer
	template<typename T>
	void safeCudaFree(T*& ptr) {
		if (ptr != nullptr) {
			cudaFree(ptr);
			ptr = nullptr;
		}
	}

public:
	// Default constructor
	CUDACachedFrame() = default;
	
	// Constructor with immediate allocation
	CUDACachedFrame(unsigned int width, unsigned int height) {
		alloc(width, height);
	}
	
	// Destructor - automatically free memory (RAII)
	~CUDACachedFrame() {
		free();
	}
	
	// Delete copy constructor and copy assignment to prevent accidental copies
	CUDACachedFrame(const CUDACachedFrame&) = delete;
	CUDACachedFrame& operator=(const CUDACachedFrame&) = delete;
	
	// Move constructor
	CUDACachedFrame(CUDACachedFrame&& other) noexcept
		: m_allocated(other.m_allocated)
		, m_width(other.m_width)
		, m_height(other.m_height)
		, d_num_valid_points(other.d_num_valid_points)
		, d_depthDownsampled(other.d_depthDownsampled)
		, d_cameraposDownsampled(other.d_cameraposDownsampled)
		, d_intensityDownsampled(other.d_intensityDownsampled)
		, d_intensityDerivsDownsampled(other.d_intensityDerivsDownsampled)
		, d_normalsDownsampled(other.d_normalsDownsampled)
	{
		// Reset other object to prevent double-free
		other.m_allocated = false;
		other.m_width = 0;
		other.m_height = 0;
		other.d_num_valid_points = nullptr;
		other.d_depthDownsampled = nullptr;
		other.d_cameraposDownsampled = nullptr;
		other.d_intensityDownsampled = nullptr;
		other.d_intensityDerivsDownsampled = nullptr;
		other.d_normalsDownsampled = nullptr;
	}
	
	// Move assignment operator
	CUDACachedFrame& operator=(CUDACachedFrame&& other) noexcept {
		if (this != &other) {
			// Free existing resources
			free();
			
			// Move data from other
			m_allocated = other.m_allocated;
			m_width = other.m_width;
			m_height = other.m_height;
			d_num_valid_points = other.d_num_valid_points;
			d_depthDownsampled = other.d_depthDownsampled;
			d_cameraposDownsampled = other.d_cameraposDownsampled;
			d_intensityDownsampled = other.d_intensityDownsampled;
			d_intensityDerivsDownsampled = other.d_intensityDerivsDownsampled;
			d_normalsDownsampled = other.d_normalsDownsampled;
			
			// Reset other object
			other.m_allocated = false;
			other.m_width = 0;
			other.m_height = 0;
			other.d_num_valid_points = nullptr;
			other.d_depthDownsampled = nullptr;
			other.d_cameraposDownsampled = nullptr;
			other.d_intensityDownsampled = nullptr;
			other.d_intensityDerivsDownsampled = nullptr;
			other.d_normalsDownsampled = nullptr;
		}
		return *this;
	}

	void alloc(unsigned int width, unsigned int height) {
		// Prevent double allocation
		if (m_allocated) {
			throw std::runtime_error("CUDACachedFrame: Already allocated. Call free() first.");
		}
		
		if (width == 0 || height == 0) {
			throw std::invalid_argument("CUDACachedFrame: Invalid dimensions");
		}
		
		try {
			// Store dimensions
			m_width = width;
			m_height = height;
			
			// Allocate memory with error checking
			CUDA_CHECK(cudaMalloc(&d_depthDownsampled, sizeof(float) * width * height));
			CUDA_CHECK(cudaMalloc(&d_cameraposDownsampled, sizeof(float4) * width * height));
			CUDA_CHECK(cudaMalloc(&d_num_valid_points, sizeof(int)));
			CUDA_CHECK(cudaMalloc(&d_intensityDownsampled, sizeof(float) * width * height));
			CUDA_CHECK(cudaMalloc(&d_intensityDerivsDownsampled, sizeof(float2) * width * height));
			CUDA_CHECK(cudaMalloc(&d_normalsDownsampled, sizeof(float4) * width * height));
			
			// Initialize memory with error checking
			CUDA_CHECK(cudaMemset(d_num_valid_points, 0, sizeof(int)));
			
			m_allocated = true;
		}
		catch (...) {
			// If any allocation fails, clean up partial allocations
			free();
			throw;
		}
	}
	
	void free() {
		if (m_allocated) {
			// Safe cleanup with error checking (non-throwing)
			safeCudaFree(d_depthDownsampled);
			safeCudaFree(d_num_valid_points);
			safeCudaFree(d_cameraposDownsampled);
			safeCudaFree(d_intensityDownsampled);
			safeCudaFree(d_intensityDerivsDownsampled);
			safeCudaFree(d_normalsDownsampled);
			
			m_allocated = false;
			m_width = 0;
			m_height = 0;
		}
	}
	
	// Query methods
	bool isAllocated() const { return m_allocated; }
	unsigned int getWidth() const { return m_width; }
	unsigned int getHeight() const { return m_height; }
	size_t getTotalMemoryUsage() const {
		if (!m_allocated) return 0;
		size_t total = 0;
		total += sizeof(float) * m_width * m_height;      // d_depthDownsampled
		total += sizeof(float4) * m_width * m_height;     // d_cameraposDownsampled
		total += sizeof(int);                             // d_num_valid_points
		total += sizeof(float) * m_width * m_height;      // d_intensityDownsampled
		total += sizeof(float2) * m_width * m_height;     // d_intensityDerivsDownsampled
		total += sizeof(float4) * m_width * m_height;     // d_normalsDownsampled
		return total;
	}

	// Member variables (kept public for compatibility with existing code)
	int* d_num_valid_points = nullptr;
	float* d_depthDownsampled = nullptr;
	float4* d_cameraposDownsampled = nullptr;
	
	// For dense color term
	float* d_intensityDownsampled = nullptr;  // This could be packed with intensityDerivatives to a float4
	float2* d_intensityDerivsDownsampled = nullptr;  // TODO: could have energy over intensity gradient instead of intensity
	float4* d_normalsDownsampled = nullptr;
};

#endif //CUDA_CACHE_UTIL
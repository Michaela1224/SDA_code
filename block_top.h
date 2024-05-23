#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;




void do_compute_top(ap_uint<128>* img_conv3_mm, 
				// conv3的权重输入
				ap_uint<128> *weight_conv3_mm, 

				// ap_uint<512>*fc_weight,  
				// bias+scalefactor
				ap_uint<128>* ddr_bias_scale_factor,  //  BIAS_BIT*16
//
				ap_uint<128>* ddr_fm_shortcut,

				ap_uint<128>* ddr_fm_back,
				ap_uint<128>* ddr_fm_shortcut_back,
				// stream<ap_uint<16 * 2> > fifo_C_deQua[16],

				const unsigned layer_bias_offset,
				const unsigned layer_weight_offset,
				// const ap_uint<4> ENCODE_MODE,
				const unsigned R,
				const unsigned C,
				const unsigned N,
				const unsigned M,
				const unsigned D,			
				const unsigned WhichPath,
				const bool CONV1_TO_MM_EN);



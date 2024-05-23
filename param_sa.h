
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include "config.h"

/*****
 * MODE控制
*/
// 计算单元控制
// bool LAYER_COMPUTE_SA_MODE;  // SA控制  FALSE为MM TRUE为CONV3 
// bool LAYER_COMPUTE_NORM_MODE; // TRUE为执行NORM FALSE为不执行
// bool LAYER_COMPUTE_QUAN_MODE; 
// bool LAYER_COMPUTE_SHORCUT_QUAN_MODE; 
// 数据加载控制
// bool LAYER_ACCESS_NORM_MODE;  // TRUE为执行NORM FALSE为不执行
// bool LAYER_ACCESS_QUAN_MODE;
// bool LAYER_ACCESS_SHORCUT_QUAN_MODE;

bool FC_INPUT_FLAG=false;
// SHORTCUT ADD
ap_uint<MAX_OUP*ILN_WIDTH*2> SHORTCUT_IN_buffer0[MAX_SHORTCUT_NORM_INBUF_LENGTH]; 
ap_uint<MAX_OUP*ILN_WIDTH*2> SHORTCUT_IN_buffer1[MAX_SHORTCUT_NORM_INBUF_LENGTH]; 

// 1D linear

// static ap_uint<FC_INP*PACK_NUM*IN_BIT> linear1d_input[MAX_FC1D_INPUT_LENGTH];
// static ap_uint<FC_INP*PACK_NUM*W_BIT> linear1d_weight[FC_OUP][MAX_FC1D_WEIGHT_LENGTH];


// static ap_int<BIAS_BIT> linear1d_bias_buffer[FC_OUP][MAX_FC_BIAS_LENGTH];
//#pragma HLS ARRAY_PARTITION variable=conv3_bias dim=1 complete

//static ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> linear1d_out_buffer[FC_OUP][MAX_FC_BIAS_LENGTH];
//#pragma HLS ARRAY_PARTITION variable=conv3_bias dim=1 complete

// static ap_int<BIAS_BIT> linear1d_out_buffer[MAX_OUP][MAX_CONV3_BIAS_LENGTH];
// //#pragma HLS ARRAY_PARTITION variable=conv3_bias dim=1 complete





// 处理resnet block的放置
static ap_uint<MAX_INP * CONV_K *W_BIT> conv3_w_buffer0[MAX_A_COL][MAX_CONV3_WEIGHT_LENGTH];   // 尽可能一次加载所有权重
//#pragma HLS ARRAY_PARTITION variable=conv3_w_buffer dim=1 complete

static ap_uint<MAX_INP * CONV_K *W_BIT> conv3_w_buffer1[MAX_A_COL][MAX_CONV3_WEIGHT_LENGTH];   // 尽可能一次加载所有权重
//#pragma HLS ARRAY_PARTITION variable=conv3_w_buffer dim=1 complete


static ap_uint<MAX_INP * IN_BIT * PACK_NUM> mm_a_buffer0[MAX_MM_FM_LENGTH];
static ap_uint<MAX_INP * IN_BIT * PACK_NUM> mm_a_buffer1[MAX_MM_FM_LENGTH];

// ap_uint<MAX_INP * W_BIT * PACK_NUM> mm_w_buffer[MAX_MM_FM_LENGTH];
// ap_uint<MAX_INP * W_BIT * PACK_NUM> mm_w_buffer1[MAX_MM_FM_LENGTH];


static ap_int<BIAS_BIT> conv3_mm_bias_buffer[MAX_OUP][MAX_CONV3_BIAS_LENGTH];
//#pragma HLS ARRAY_PARTITION variable=conv3_bias dim=1 complete



static ap_uint<128> scale_factor_buffer;   // [0]:conv3 [1] fc


	// scale_factor_buffer[0]=temp_scale_factor(15,0);
	// scale_factor_buffer[1]=temp_scale_factor(31,16);
	// scale_factor_buffer[2]=temp_scale_factor(47,32);
	// scale_factor_buffer[3]=temp_scale_factor(63,48);
	// scale_factor_buffer[4]=temp_scale_factor(79,64);
	// scale_factor_buffer[5]=temp_scale_factor(95,80);




static LN_BIAS_DB ln_gamma_buffer[MAX_NORM_PE][MAX_NORM_BIAS_LENGTH];

static LN_BIAS_DB ln_beta_buffer[MAX_NORM_PE][MAX_NORM_BIAS_LENGTH];


// mm: MAX_M_LENGTH: M/MAX_OUP
// conv: MAX_M_LENGTH: M/(MAX_OUP*2)
static ap_uint<LN_PWF_FACTOR_BIT> ln_ptf_factor_buffer0[MAX_OUP][MAX_NORM_PWF_FACTOR_LENGTH];
static ap_uint<LN_PWF_FACTOR_BIT> ln_ptf_factor_buffer1[MAX_OUP][MAX_NORM_PWF_FACTOR_LENGTH];

static ap_uint<ILN_WIDTH*2*2> LN_IN_buffer0[MAX_OUP][MAX_NORM_INBUF_LENGTH]; 
static ap_uint<ILN_WIDTH*2*2> LN_IN_buffer1[MAX_OUP][MAX_NORM_INBUF_LENGTH]; 



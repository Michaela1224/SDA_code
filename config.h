


#define CONV_K 3


#define MAX_INP 20
#define MAX_OUP 10

#define SA_INP 5
#define SA_OUP 5

#define MAX_A_ROW MAX_INP/SA_INP
#define MAX_A_COL MAX_OUP/SA_OUP 

// 假定 S2三个处理的并行度都相同
#define MAX_NORM_PE 5  // Norm单元的并行度
#define MAX_SOFTMAX_STAGE1_PE 5
#define MAX_SOFTMAX_STAGE2_PE 5
#define MAX_GELU_PE 5



#define FC_INP 2
#define FC_OUP 10  // 02-24-setting true
#define LINREAR_N 1280


#define IN_BIT 8
#define W_BIT 4 
#define PACK_NUM 2 
#define PACK_CONV_NUM 3 
#define PACK_OUT_NUM 4 
#define ACC_BIT 36   // 02-23-setting true
#define BIAS_BIT 16
typedef ap_fixed<16, 8> LN_BIAS_DB;  // LN gamma、beta
#define OUT_BIT 8

#define SHORTCUT_BIT 8
#define DEQUAN_BIT 16   
#define DEQUAN_INTEGER_BIT 8   
#define Shift_Num 17

typedef ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> De_Quan_DB;  // LN gamma、beta

#define QUAN_FACTOR_BIT 16
#define QUAN_FACTOR_INTEGER_BIT 8
typedef ap_fixed<QUAN_FACTOR_BIT, QUAN_FACTOR_INTEGER_BIT> Quan_Factor_DB;  // LN gamma、beta

#define SILU_BIT 16
#define SILU_INTEGER_BIT 8
typedef ap_fixed<SILU_BIT, SILU_INTEGER_BIT> SILU_DB;  // LN gamma、beta




#define ILN_WIDTH 8
#define LN_PWF_FACTOR_BIT 2   // LN PWF FACTOR
#define ILN_N_MEAN_WIDTH  27// LN的输入 8+3+17 约为
#define ILN_N_VAR_WIDTH  35// LN的输入 8+4+6 +17= 35
#define ILN_MEAN_WIDTH  11// LN的输入 8+3
#define ILN_VAR_WIDTH  18// LN的输入 8+4+6
#define ILN_OUT_WIDTH 18 // LN的输入
#define ILN_OUT_INTEGER_WIDTH 12 // LN的输入
typedef ap_fixed<ILN_OUT_WIDTH, ILN_OUT_INTEGER_WIDTH> LN_OUT_DB; 



#define SOFTMAX_OUT_WIDTH 16
#define SOFTMAX_OUT_INTEGER_WIDTH 2 // LN的输入



#define MAX_SOFTMAX_M_LENGTH 4100  // 02-24-setting true
typedef ap_fixed<24, 8> SOFTMAX_SUM_DB; 



#define GELU_OUT_WIDTH 16 // LN的输入
#define GELU_OUT_INTEGER_WIDTH 8 // LN的输入


#define MAX_SOFTMAX_INBUF_LENGTH 8240 // 02-23-setting true
#define MAX_GELU_INBUF_LENGTH 8240     // 02-23-setting true
#define MAX_GELU_ROW_INBUF_LENGTH 5120     // 02-23-setting true
#define MAX_NORM_INBUF_LENGTH 8240      // 02-23-setting true
#define MAX_SHORTCUT_NORM_INBUF_LENGTH 2560      // 02-23-setting true

#define AXI_BIAS_BIT 512

// #define MAX_M 320
// #define MAX_N 320

#define MAX_CONV3_WEIGHT_LENGTH  7680  // 02-23-setting true
#define MAX_CONV3_BIAS_LENGTH    410      // 02-23-setting true 10240/10
#define MAX_SCALE_FACTOR_LENGTH 6 //quan/dequan number
#define MAX_MM_FM_LENGTH 5120  // 02-23-setting true
#define MAX_NORM_BIAS_LENGTH 256     // 02-23-setting true
#define MAX_NORM_PWF_FACTOR_LENGTH 128     // 02-23-setting true

// #define MAX_FC1D_WEIGHT_LENGTH 1600     // (1280/(FC_INP*2))*(1280/FC_OUP)  02-24-setting true
// #define MAX_FC1D_INPUT_LENGTH 320     // 02-23-setting true
// #define MAX_FC_BIAS_LENGTH 640 // 02-23-setting true

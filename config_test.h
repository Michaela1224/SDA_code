// MM
#define R 160
#define N 160
#define M 160

// WS


#define CONV_R 32
#define CONV_C 32
#define CONV_N 80
#define CONV_M 80
#define CONV_D 20

#define layer1_W_offset 0
#define layer2_W_offset ((CONV_K*CONV_N)/MAX_INP)*CONV_M*2

#define layer1_BIAS_offset 0
#define layer2_BIAS_offset (CONV_M/(MAX_OUP/2))+1+(CONV_M/(2*MAX_OUP))

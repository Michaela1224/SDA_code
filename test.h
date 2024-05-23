#include <ap_int.h>
#include <hls_stream.h>
#include "config.h"
#include "config_test.h"
#define GLODEN_DEBUG
using namespace std;



void generate_mm_shortcut_128btest(ap_uint<128> *mm_ddr_shortcut){

    unsigned seed=0;
	srand(seed);
    
    ap_uint<MAX_OUP*IN_BIT*2> a;
    ap_uint<16> a_16b;
    ap_uint<MAX_OUP*IN_BIT*2/2> temp_80b0,temp_80b1;
    ap_uint<128> temp_128b0,temp_128b1;
    int cnt=0;
	for(int i=0; i<R*M/(2*MAX_OUP); i++){
        
        for(int j=0;j<MAX_OUP;j++){

            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(MAX_OUP*IN_BIT*2-1,MAX_OUP*IN_BIT*2-IN_BIT*2)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        (temp_80b1,temp_80b0)=a;
        temp_128b0=temp_80b0;
        temp_128b1=temp_80b1;

        mm_ddr_shortcut[cnt]=temp_128b0;
        cnt++;

        mm_ddr_shortcut[cnt]=temp_128b1;
        cnt++;

        a=0;


	}

}


void generate_mm_shortcut_512btest(ap_uint<512> *mm_ddr_shortcut){

    ap_uint<MAX_OUP*IN_BIT*2> a;
    ap_uint<16> a_16b;
	for(int i=0; i<R*M/(2 * MAX_OUP); i++){
        
        for(int j=0;j<MAX_OUP;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(MAX_OUP*ILN_WIDTH*2-1,ILN_WIDTH*2)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        mm_ddr_shortcut[i]=(ap_uint<512>)a;
        a=0;


	}

}

// conv-shortcut-output
void generate_conv_shortcut_512btest(ap_uint<512> *conv_ddr_fcvu){

    ap_uint<MAX_OUP*IN_BIT*2> a;
    ap_uint<16> a_16b;
	for(int i=0; i<CONV_R*CONV_C*CONV_M/(2 * MAX_OUP); i++){
        
        for(int j=0;j<MAX_OUP;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(MAX_OUP*ILN_WIDTH*2-1,ILN_WIDTH*2)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        conv_ddr_fcvu[i]=(ap_uint<512>)a;
        a=0;


	}

}


// conv-fcvu-output

void generate_conv_fcvu_512btest(ap_uint<512> *conv_ddr_fcvu){

    ap_uint<MAX_OUP*ILN_WIDTH*2> a;
    ap_uint<16> a_16b;
	for(int i=0; i<CONV_M/MAX_OUP; i++){
        
        for(int j=0;j<MAX_OUP;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(MAX_OUP*ILN_WIDTH*2-1,ILN_WIDTH*2)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        conv_ddr_fcvu[i]=(ap_uint<512>)a;
        a=0;


	}

}


// ln------parameter


void generate_ln_bias(LN_BIAS_DB *INPUT_GaMMA, LN_BIAS_DB *INPUT_BeTa, int NUM){
    unsigned seed=0;
	srand(seed);

	FILE* fp_true_res0 = fopen("ln_gama.txt", "wb");
	FILE* fp_true_res1 = fopen("ln_beta.txt", "wb");
	LN_BIAS_DB fp_temp0,fp_temp1;
	for(int i=0; i<NUM; i++){

		fp_temp0=(rand()/(float)RAND_MAX)*127-(rand()/(float)RAND_MAX)*127;
		// cout<<"fp_temp:"<<fp_temp<<endl;
		INPUT_GaMMA[i]=fp_temp0;
		// cout <<"The Value of A0: \t" <<INPUT_GaMMA[i] << "\t Binary format: \t" <<INPUT_GaMMA[i].to_string(2).c_str()<< '\n';
		fprintf(fp_true_res0, "%lf\n", (double)fp_temp0);

		fp_temp1=(rand()/(float)RAND_MAX)*127-(rand()/(float)RAND_MAX)*127;
		// cout<<"fp_temp:"<<fp_temp<<endl;
		INPUT_BeTa[i]=fp_temp1;
		fprintf(fp_true_res1, "%lf\n", (double)fp_temp1);


	}
	fclose(fp_true_res0);
	fclose(fp_true_res1);
}

void generate_conv_ln_ptf_factor(ap_uint<8> ptf_factor[CONV_M]){

	unsigned seed=0;
	srand(seed);

	FILE* fp_true_A = fopen("ln_conv_ptf_factor.txt", "wb");
	ap_uint<2> temp_2bit;
	ap_uint<8> temp_8bit;

	for (int j = 0; j < CONV_M/4; j++) {
		for (int r = 0; r < 4; r++) {
			
			temp_2bit = (ap_uint<LN_PWF_FACTOR_BIT>)rand();
			temp_8bit((r+1)*2-1,r*2)=temp_2bit;
			//   std::cout <<"The Value of A["<<r<<"]["<<n<<"]: \t" <<A[r][n] << " \t Binary format: \t" <<A[r][n].to_string(2).c_str()<< '\n';
			fprintf(fp_true_A, "%d\n", (int)temp_2bit);
		}
		ptf_factor[j]=temp_8bit;
	}

	fclose(fp_true_A);

}


void generate_mm_ln_ptf_factor(ap_uint<8> ptf_factor[M]){

	unsigned seed=0;
	srand(seed);

	FILE* fp_true_A = fopen("ln_mm_ptf_factor.txt", "wb");
	ap_uint<2> temp_2bit;
	ap_uint<8> temp_8bit;

	for (int j = 0; j < M/4; j++) {
		for (int r = 0; r < 4; r++) {
			
			temp_2bit = (ap_uint<LN_PWF_FACTOR_BIT>)rand();
			temp_8bit((r+1)*2-1,r*2)=temp_2bit;
			//   std::cout <<"The Value of A["<<r<<"]["<<n<<"]: \t" <<A[r][n] << " \t Binary format: \t" <<A[r][n].to_string(2).c_str()<< '\n';
			fprintf(fp_true_A, "%d\n", (int)temp_2bit);
		}
		ptf_factor[j]=temp_8bit;
	}

	fclose(fp_true_A);

}


void generate_conv_bias(ap_int<BIAS_BIT> bias[CONV_M/CONV_D][CONV_D/MAX_OUP][MAX_OUP]){

	unsigned seed=0;
	srand(seed);
    int cnt=0;
	FILE* fp_true_bias = fopen("true_bias.txt", "wb");
    for (int n=0; n<CONV_M/CONV_D; n++) {
        for (int k=0; k<CONV_D/MAX_OUP; k++) {
            for(int m=0; m<MAX_OUP; m++){
                // bias[n][k][m] = (ap_int<BIAS_BIT>)rand()-(ap_int<BIAS_BIT>)rand();
                bias[n][k][m] = cnt;
                //   std::cout <<"The Value of W["<<n<<"]["<<m<<"]: \t" <<A[n][m] << " \t Binary format: \t" <<A[n][m].to_string(2).c_str()<< '\n';
                fprintf(fp_true_bias, "%d", (int)bias[n][k][m]);
                cnt++;
            }
        }
		// fprintf(fp_true_bias, "\n");
    }

	fclose(fp_true_bias);
    
}


void generate_mm_bias(ap_int<BIAS_BIT> bias[M/(2*MAX_OUP)][2][MAX_OUP]){

	unsigned seed=0;
	srand(seed);
    int cnt=0;
	FILE* fp_true_bias = fopen("true_mm_bias.txt", "wb");
    for (int n=0; n<CONV_M/(2*MAX_OUP); n++) {
        for (int k=0; k<2; k++) {
            for(int m=0; m<MAX_OUP; m++){
                // bias[n][k][m] = (ap_int<BIAS_BIT>)rand()-(ap_int<BIAS_BIT>)rand();
                bias[n][k][m] = cnt;
                //   std::cout <<"The Value of W["<<n<<"]["<<m<<"]: \t" <<A[n][m] << " \t Binary format: \t" <<A[n][m].to_string(2).c_str()<< '\n';
                fprintf(fp_true_bias, "%d", (int)bias[n][k][m]);
                cnt++;
            }
        }
		// fprintf(fp_true_bias, "\n");
    }

	fclose(fp_true_bias);
    
}


void generate_conv_norm_factor(ap_uint<256> *input_ln_parameter){

	ap_int<BIAS_BIT> conv3_bias[CONV_M/CONV_D][CONV_D/MAX_OUP][MAX_OUP];

	generate_conv_bias(conv3_bias);

	ap_int<BIAS_BIT> conv3_scale_factor=ap_int<BIAS_BIT>(1234);


	LN_BIAS_DB *input_ln_gamma = (LN_BIAS_DB *)malloc(CONV_M*sizeof(LN_BIAS_DB) );
	// LN_BIAS_DB input_gamma[CONV_M];

	LN_BIAS_DB *input_ln_beta = (LN_BIAS_DB *)malloc(CONV_M*sizeof(LN_BIAS_DB) );

	generate_ln_bias(input_ln_gamma,input_ln_beta,CONV_M);


	ap_uint<8> conv_ln_ptf_factor[CONV_M/4];


	generate_conv_ln_ptf_factor(conv_ln_ptf_factor);


	


	memcpy(input_ln_parameter, conv3_bias, (CONV_M)*2);
	
	input_ln_parameter[CONV_M/16]=conv3_scale_factor;
    cout<<input_ln_parameter[CONV_M/16]<<endl;

	memcpy(input_ln_parameter+(CONV_M/16)+1, input_ln_gamma,CONV_M*2);
	free(input_ln_gamma);

	memcpy(input_ln_parameter+(CONV_M/16)+1+(CONV_M/16), input_ln_beta,CONV_M*2);
	free(input_ln_beta);

	memcpy(input_ln_parameter+(CONV_M/16)+1+(CONV_M/16)*2, conv_ln_ptf_factor,M/4);	

}






void generate_mm_allbias_test(ap_uint<AXI_BIAS_BIT> *mm_ddr_bias){

    ap_uint<256> a;
    ap_uint<16> a_16b;
	for(int i=0; i<23; i++){
        
        for(int j=0;j<16;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(256-1,240)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        mm_ddr_bias[i]=a;
        a=0;


	}

}



void generate_conv_allbias_128btest(ap_uint<128> *mm_ddr_bias, unsigned num0, unsigned numlines){


	unsigned seed=0;
	srand(seed);
    
    ap_uint<128> a;
    ap_uint<16> a_16b;
	for(int i=0; i<num0; i++){
        
        for(int j=0;j<8;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(128-1,112)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        mm_ddr_bias[i]=(ap_uint<128>)a;
        a=0;
	}


	ap_uint<BIAS_BIT> conv3_scale_factor=ap_uint<BIAS_BIT>(275);
    ap_uint<BIAS_BIT> integer_conv3_scale_factor;
    integer_conv3_scale_factor(BIAS_BIT-1,0)=conv3_scale_factor(BIAS_BIT-1,0);


    ap_fixed<16, 8> quan_factor=41.2345;
    ap_uint<BIAS_BIT> integer_quan_factor;
    integer_quan_factor(BIAS_BIT-1,0)=quan_factor(BIAS_BIT-1,0);

    cout<<"quan_factor:"<<quan_factor<<endl;
	
    ap_fixed<16, 8> short_quan_factor=32.2345;
    ap_uint<BIAS_BIT> integer_short_quan_factor;
    integer_short_quan_factor(BIAS_BIT-1,0)=short_quan_factor(BIAS_BIT-1,0);

    cout<<"short_quan_factor:"<<short_quan_factor<<endl;

	ap_uint<BIAS_BIT> fc_scale_factor=ap_uint<BIAS_BIT>(200);
    ap_uint<BIAS_BIT> integer_fc_scale_factor;
    integer_fc_scale_factor(BIAS_BIT-1,0)=fc_scale_factor(BIAS_BIT-1,0);

    ap_fixed<16, 8> short_dequan_factor=32.2345;
    ap_uint<BIAS_BIT> integer_short_dequan_factor;
    integer_short_dequan_factor(BIAS_BIT-1,0)=short_dequan_factor(BIAS_BIT-1,0);

    ap_uint<BIAS_BIT*5> integer_factor;
    integer_factor=(integer_short_dequan_factor, fc_scale_factor, integer_short_quan_factor,integer_quan_factor,integer_conv3_scale_factor);

    mm_ddr_bias[num0]=(ap_uint<128>)integer_factor;


	for(int i=0; i<numlines-num0-1; i++){
        
        for(int j=0;j<8;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(128-1,112)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        mm_ddr_bias[num0+1+i]=(ap_uint<128>)a;
        a=0;
	}



}


void generate_conv_allbias_512btest(ap_uint<AXI_BIAS_BIT> *mm_ddr_bias, unsigned num0, unsigned numlines){


	unsigned seed=0;
	srand(seed);
    
    ap_uint<512> a;
    ap_uint<16> a_16b;
	for(int i=0; i<num0; i++){
        
        for(int j=0;j<32;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(512-1,496)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        mm_ddr_bias[i]=(ap_uint<512>)a;
        a=0;
	}


	ap_uint<BIAS_BIT> conv3_scale_factor=ap_uint<BIAS_BIT>(275);
    ap_uint<BIAS_BIT> integer_conv3_scale_factor;
    integer_conv3_scale_factor(BIAS_BIT-1,0)=conv3_scale_factor(BIAS_BIT-1,0);


    ap_fixed<16, 8> quan_factor=41.2345;
    ap_uint<BIAS_BIT> integer_quan_factor;
    integer_quan_factor(BIAS_BIT-1,0)=quan_factor(BIAS_BIT-1,0);

    cout<<"quan_factor:"<<quan_factor<<endl;
	
    ap_fixed<16, 8> short_quan_factor=32.2345;
    ap_uint<BIAS_BIT> integer_short_quan_factor;
    integer_short_quan_factor(BIAS_BIT-1,0)=short_quan_factor(BIAS_BIT-1,0);

    cout<<"short_quan_factor:"<<short_quan_factor<<endl;

	ap_uint<BIAS_BIT> fc_scale_factor=ap_uint<BIAS_BIT>(200);
    ap_uint<BIAS_BIT> integer_fc_scale_factor;
    integer_fc_scale_factor(BIAS_BIT-1,0)=fc_scale_factor(BIAS_BIT-1,0);

    ap_fixed<16, 8> short_dequan_factor=32.2345;
    ap_uint<BIAS_BIT> integer_short_dequan_factor;
    integer_short_dequan_factor(BIAS_BIT-1,0)=short_dequan_factor(BIAS_BIT-1,0);

    ap_uint<BIAS_BIT*5> integer_factor;
    integer_factor=(integer_short_dequan_factor, fc_scale_factor, integer_short_quan_factor,integer_quan_factor,integer_conv3_scale_factor);

    mm_ddr_bias[num0]=(ap_uint<512>)integer_factor;


	for(int i=0; i<numlines-num0-1; i++){
        
        for(int j=0;j<32;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(512-1,496)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        mm_ddr_bias[num0+1+i]=(ap_uint<512>)a;
        a=0;
	}



}


void generate_mm_allbias_512btest(ap_uint<AXI_BIAS_BIT> *mm_ddr_bias){


	unsigned seed=0;
	srand(seed);

    ap_uint<320> a;
    ap_uint<16> a_16b;
	for(int i=0; i<23; i++){
        
        for(int j=0;j<20;j++){
            a=a>>16;
            a_16b=rand();
            // cout <<"The Value of Var_p: \t" <<a_16b<< " \t Binary format: \t" <<a_16b.to_string(2).c_str()<< '\n';
            a(320-1,304)=a_16b;
            
        }        
        // cout <<"The Value of Var_p: \t" <<a<< " \t Binary format: \t" <<a.to_string(2).c_str()<< '\n';
        mm_ddr_bias[i]=(ap_uint<512>)a;
        a=0;


	}

}




////////////////////////////

void generate_mm_a(ap_int<IN_BIT> A[R][N]){

	unsigned seed=0;
	srand(seed);

	FILE* fp_true_A = fopen("true_A.txt", "wb");

    for (int r = 0; r < R; r++) {
		for(int n=0; n<N; n++){
			A[r][n] = (ap_int<IN_BIT>)rand()-(ap_int<IN_BIT>)rand();
			//   std::cout <<"The Value of A["<<r<<"]["<<n<<"]: \t" <<A[r][n] << " \t Binary format: \t" <<A[r][n].to_string(2).c_str()<< '\n';
			fprintf(fp_true_A, "%d\t", (int)A[r][n]);
		}
		fprintf(fp_true_A, "\n");
    }

	fclose(fp_true_A);

}


void generate_mm_w(ap_int<W_BIT> W[N][M]){

	unsigned seed=0;
	srand(seed);

	FILE* fp_true_W = fopen("true_W.txt", "wb");
    for (int n=0; n<N; n++) {
		for(int m=0; m<M; m++){
			W[n][m] = (ap_int<W_BIT>)rand()-(ap_int<W_BIT>)rand();
			//   std::cout <<"The Value of W["<<n<<"]["<<m<<"]: \t" <<A[n][m] << " \t Binary format: \t" <<A[n][m].to_string(2).c_str()<< '\n';
			fprintf(fp_true_W, "%d\t", (int)W[n][m]);
		}
		fprintf(fp_true_W, "\n");
    }

	fclose(fp_true_W);
    
}



void generate_conv3_a(ap_int<IN_BIT> conv3_A[CONV_R][CONV_C][CONV_N]){

	unsigned seed=0;
	srand(seed);

	FILE* fp_true_A = fopen("true_conv3_A.txt", "wb");
    for (int r=0; r<CONV_R; r++) {
		for(int c=0; c<CONV_C; c++){    
            for (int n=0; n<CONV_N; n++) {
			            conv3_A[r][c][n] = (ap_int<IN_BIT>)rand()-(ap_int<IN_BIT>)rand();
			//   std::cout <<"The Value of W["<<n<<"]["<<m<<"]: \t" <<A[n][m] << " \t Binary format: \t" <<A[n][m].to_string(2).c_str()<< '\n';
			        fprintf(fp_true_A, "%d\n", (int)conv3_A[r][c][n]);
            }
        }
    }

	fclose(fp_true_A);
    
}


void generate_conv3_weight(ap_int<W_BIT> conv3_weight[CONV_K][CONV_K][CONV_N][CONV_M]){

	unsigned seed=0;
	srand(seed);

	FILE* fp_true_W = fopen("true_conv3_W.txt", "wb");
    for (int kr=0; kr<CONV_K; kr++) {
		for(int kc=0; kc<CONV_K; kc++){    
            for (int n=0; n<CONV_N; n++) {
                for(int m=0; m<CONV_M; m++){
			            conv3_weight[kr][kc][n][m] = (ap_int<W_BIT>)rand()-(ap_int<W_BIT>)rand();
			//   std::cout <<"The Value of W["<<n<<"]["<<m<<"]: \t" <<A[n][m] << " \t Binary format: \t" <<A[n][m].to_string(2).c_str()<< '\n';
			        fprintf(fp_true_W, "%d\n", (int)conv3_weight[kr][kc][n][m]);
		        }
            }
        }
    }

	fclose(fp_true_W);
    
}



void reorg_conv3_a(ap_int<IN_BIT> conv3_A[CONV_R][CONV_C][CONV_N],
            ap_uint<MAX_INP * PACK_NUM * IN_BIT> *conv3_ddr_a){

int cnt=0;
ap_int<MAX_INP * PACK_NUM * IN_BIT> temp;
    for (int r=0; r<CONV_R; r++) {
        for (int n=0; n<CONV_N/MAX_INP; n++) {
		    for(int c=0; c<CONV_C/2; c++){    
                for(int x=0;x<MAX_INP;x++){
                    for(int s=0;s<PACK_NUM;s++){
                        temp=temp>>IN_BIT;
                        temp(MAX_INP * PACK_NUM * IN_BIT-1,(MAX_INP * PACK_NUM-1) * IN_BIT)=conv3_A[r][c*PACK_NUM+s][n*MAX_INP+x];
                    }
                }
                conv3_ddr_a[cnt]=temp;
                cnt++;
            }
        }
    }

}

// 
void reorg_conv3_weight(ap_int<W_BIT> conv3_weight[CONV_K][CONV_K][CONV_N][CONV_M],
            ap_uint<256> packing_conv3_weight[((CONV_K*CONV_N)/MAX_INP)*(CONV_M/(MAX_A_COL))][MAX_A_COL]){

ap_uint<MAX_INP * CONV_K * W_BIT> packing_conv3_tmp[((CONV_K*CONV_N)/MAX_INP)*(CONV_M/MAX_OUP)][MAX_OUP];
ap_uint<MAX_INP * CONV_K * W_BIT> tmp;
ap_uint<256> tmp_256;

    // cout<<((CONV_K*CONV_N)/MAX_INP)*(CONV_M/MAX_OUP)<<endl;

    for(int m=0; m<CONV_M/MAX_OUP; m++){
        for(int y=0; y<MAX_OUP; y++){
            for (int kr=0; kr<CONV_K; kr++) {
                for (int n=0; n<CONV_N/MAX_INP; n++) {
                    for(int x=0; x<MAX_INP; x++){
                        for(int kc=0; kc<CONV_K; kc++){ 
                            tmp=tmp>>W_BIT;
			                tmp(MAX_INP * CONV_K * W_BIT-1,(MAX_INP * CONV_K-1) * W_BIT)=conv3_weight[kr][kc][n*MAX_INP+x][m*MAX_OUP+y];
                        }
                    }
                    packing_conv3_tmp[m*CONV_K*(CONV_N/MAX_INP)+kr*(CONV_N/MAX_INP)+n][y]=tmp;
                    // cout<<"d1: "<<y<<"   d0:"<<m*CONV_K*(CONV_N/MAX_INP)+kr*(CONV_N/MAX_INP)+n<<endl;
                    
                }
            }
        }
    }


    for(int m=0; m<((CONV_K*CONV_N)/MAX_INP)*(CONV_M/MAX_OUP); m++){
        // cout<<MAX_OUP/(MAX_A_COL)<<endl;
        for(int y=0; y<MAX_OUP/(MAX_A_COL); y++){
            for(int s=0; s<MAX_A_COL;s++){
                tmp=packing_conv3_tmp[m][s*(MAX_OUP/(MAX_A_COL))+y];
                // cout<<tmp<<endl;
                tmp_256=tmp;
                packing_conv3_weight[m*(MAX_OUP/(MAX_A_COL))+y][s]=tmp_256;
            }
        }

    }

    
}









void generate_mm_output(ap_int<IN_BIT> A[R][N], ap_int<W_BIT> W[N][M],float O_golden[R][M]){

	FILE* fp_true_res = fopen("gloden_out.txt", "wb");
	for (int r = 0; r < R; r++){
		for (int m=0; m<M; m++) {
			O_golden[r][m] = 0;
			for (int n=0; n<N; n++) {
				O_golden[r][m] = O_golden[r][m] + A[r][n] * W[n][m];
			}
			fprintf(fp_true_res, "%d\t", (int)O_golden[r][m]);
		}
		fprintf(fp_true_res, "\n");
	}
	fclose(fp_true_res);
}


bool loadFile_txt_image(const char* name1,  ap_uint<MAX_INP * PACK_NUM * IN_BIT> *conv3_ddr_to){

	FILE* fp1 = fopen(name1, "rb");
	int i = 0;
	int j = 0;
	ap_int<4> tmp;  //输入为3bit
	ap_uint<MAX_INP * PACK_NUM * IN_BIT> in_data;

	int rep;
	int temp;
	FILE* fp_input = fopen("input_verfy.txt", "wb");
	if ((fp1 == NULL)) {
		std::cout << "Load Error!" << std::endl;
		return false;
	}


	for (i = 0; i<CONV_R*CONV_C*CONV_N/(MAX_INP * PACK_NUM); i++) {
		for (j = 0; j<MAX_INP * PACK_NUM; j++) {
			fscanf(fp1, "%d", &temp);  //数据格式为double
			
			tmp = (ap_int<4>)temp;  //数据转化为输入bit
			fprintf(fp_input, "%d\n", int(tmp));
//			cout << "start:" << tmp <<endl;
			in_data(IN_BIT*(j+1)-1, IN_BIT*j) = tmp;
		}
		conv3_ddr_to[i] = in_data;
	}

	fflush(fp1);  //清除读写缓冲区。强迫将缓冲区的数据写回参数stream指定的文件中
	fclose(fp1);
	fclose(fp_input);
	std::cout << "Load Success!" << std::endl;
	return true;
}


void host_DDR_A(ap_int<IN_BIT> A_from[R][N], ap_uint<MAX_INP * PACK_NUM * IN_BIT> *DDR_A_to){
  /* Variable Declaration */
  ap_uint<MAX_INP * PACK_NUM * IN_BIT> packing_in;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
    //   FILE* fpa = fopen("a_stream_pe00_gold.txt", "wb");
    FILE* fpa = fopen("a_stream_pe1x_gold.txt", "wb");
#endif
    int cnt=0;
    for(int r=0;r<R/(MAX_INP*2);r++){
        for(int n=0;n<N;n++){
            for(int x=0;x<MAX_INP;x++){
                for(int sr=0;sr<PACK_NUM;sr++){
                    packing_in=packing_in>>IN_BIT;
                    packing_in(MAX_INP * PACK_NUM * IN_BIT-1,(MAX_INP * PACK_NUM-1) * IN_BIT)=A_from[r*MAX_INP*PACK_NUM+x*PACK_NUM+sr][n];
                    
                    #ifdef GLODEN_DEBUG
                        if(x==1){
                            ap_int<IN_BIT> a0=A_from[r*MAX_INP*PACK_NUM+x*PACK_NUM+sr][n];

                            fprintf(fpa, "%d\n", (int)a0);
                        }

                    #endif
                }
            }
            DDR_A_to[cnt]=packing_in;
            cnt++;
        }
    }

#ifdef GLODEN_DEBUG
      fclose(fpa);
#endif

}



void host_DDR_A_512b(ap_int<IN_BIT> A_from[R][N], ap_uint<512> *DDR_A_to){
  /* Variable Declaration */
  ap_uint<MAX_INP * PACK_NUM * IN_BIT> packing_in;
  ap_uint<512> packing_in_512b;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
    //   FILE* fpa = fopen("a_stream_pe00_gold.txt", "wb");
    FILE* fpa = fopen("a_stream_pe1x_gold.txt", "wb");
#endif
    int cnt=0;
    for(int r=0;r<R/(MAX_INP*2);r++){
        for(int n=0;n<N;n++){
            for(int x=0;x<MAX_INP;x++){
                for(int sr=0;sr<PACK_NUM;sr++){
                    packing_in=packing_in>>IN_BIT;
                    packing_in(MAX_INP * PACK_NUM * IN_BIT-1,(MAX_INP * PACK_NUM-1) * IN_BIT)=A_from[r*MAX_INP*PACK_NUM+x*PACK_NUM+sr][n];
                    
                    #ifdef GLODEN_DEBUG
                        if(x==1){
                            ap_int<IN_BIT> a0=A_from[r*MAX_INP*PACK_NUM+x*PACK_NUM+sr][n];

                            fprintf(fpa, "%d\n", (int)a0);
                        }

                    #endif
                }
            }
            packing_in_512b=((ap_uint<512 - MAX_INP * PACK_NUM * IN_BIT>)0,packing_in);
            DDR_A_to[cnt]=packing_in_512b;
            cnt++;
        }
    }

#ifdef GLODEN_DEBUG
      fclose(fpa);
#endif

}


void host_DDR_A_128b(ap_int<IN_BIT> A_from[R][N], ap_uint<128> *DDR_A_to){
  /* Variable Declaration */
  ap_uint<MAX_INP * PACK_NUM * IN_BIT> packing_in;
  ap_uint<384> packing_in_384b;
  ap_uint<128> packing_in_128b0,packing_in_128b1,packing_in_128b2;
  
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
    //   FILE* fpa = fopen("a_stream_pe00_gold.txt", "wb");
    FILE* fpa = fopen("a_stream_pe1x_gold.txt", "wb");
#endif
    int cnt=0;
    for(int r=0;r<R/(MAX_INP*2);r++){
        for(int n=0;n<N;n++){
            for(int x=0;x<MAX_INP;x++){
                for(int sr=0;sr<PACK_NUM;sr++){
                    packing_in=packing_in>>IN_BIT;
                    packing_in(MAX_INP * PACK_NUM * IN_BIT-1,(MAX_INP * PACK_NUM-1) * IN_BIT)=A_from[r*MAX_INP*PACK_NUM+x*PACK_NUM+sr][n];
                    
                    #ifdef GLODEN_DEBUG
                        if(x==1){
                            ap_int<IN_BIT> a0=A_from[r*MAX_INP*PACK_NUM+x*PACK_NUM+sr][n];

                            fprintf(fpa, "%d\n", (int)a0);
                        }

                    #endif
                }
            }
            packing_in_384b=((ap_uint<384 - MAX_INP * PACK_NUM * IN_BIT>)0,packing_in);
            (packing_in_128b2,packing_in_128b1,packing_in_128b0)=packing_in_384b;

            DDR_A_to[cnt]=packing_in_128b0;
            cnt++;

            DDR_A_to[cnt]=packing_in_128b1;
            cnt++;

            DDR_A_to[cnt]=packing_in_128b2;
            cnt++;

        }
    }

#ifdef GLODEN_DEBUG
      fclose(fpa);
#endif

}



void host_DDR_W_Softmax(ap_int<W_BIT> W_from[N][M], ap_uint<MAX_OUP * PACK_NUM * W_BIT>*DDR_W_to){
  /* Variable Declaration */
  ap_uint<MAX_INP * PACK_NUM * W_BIT> packing_in;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
      FILE* fpw  = fopen("w_stream_pex1_gold.txt", "wb");
#endif
  int cnt=0;

    for(int rep=0;rep<2;rep++){
        for(int m=0;m<M/(MAX_OUP*2);m++){
            for(int n=0;n<N;n++){
                for(int y=0;y<MAX_OUP;y++){
                    for(int sm=0;sm<PACK_NUM;sm++){
                        packing_in=packing_in>>W_BIT;
                        packing_in(MAX_INP * PACK_NUM * W_BIT-1,(MAX_INP * PACK_NUM-1) * W_BIT)=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];


                        #ifdef GLODEN_DEBUG
                        if(y==1){
                            ap_int<W_BIT> w0=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];

                            fprintf(fpw, "%d\n", (int)w0);
                        }
                        #endif

                    }
                }
                DDR_W_to[cnt]=packing_in;
                cnt++;
            }
        }
    }


#ifdef GLODEN_DEBUG
      fclose(fpw);
#endif

}


void host_DDR_W_Softmax_128b(ap_int<W_BIT> W_from[N][M], ap_uint<128> *DDR_W_to){
  /* Variable Declaration */
  ap_uint<MAX_OUP * PACK_NUM * W_BIT> packing_in;
  ap_uint<128> packing_in_256b;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
      FILE* fpw  = fopen("w_stream_pex1_gold.txt", "wb");
#endif
  int cnt=0;

    for(int rep=0;rep<2;rep++){
        for(int m=0;m<M/(MAX_OUP*2);m++){
            for(int n=0;n<N;n++){
                for(int y=0;y<MAX_OUP;y++){
                    for(int sm=0;sm<PACK_NUM;sm++){
                        packing_in=packing_in>>W_BIT;
                        packing_in(MAX_OUP * PACK_NUM * W_BIT-1,(MAX_OUP * PACK_NUM-1) * W_BIT)=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];


                        #ifdef GLODEN_DEBUG
                        if(y==1){
                            ap_int<W_BIT> w0=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];

                            fprintf(fpw, "%d\n", (int)w0);
                        }
                        #endif

                    }
                }
            packing_in_256b=((ap_uint<128 - MAX_OUP * PACK_NUM * W_BIT>)0, packing_in);
            DDR_W_to[cnt]=packing_in;
            cnt++;
            }
        }
    }


#ifdef GLODEN_DEBUG
      fclose(fpw);
#endif

}

void host_DDR_W_Softmax_256b(ap_int<W_BIT> W_from[N][M], ap_uint<256> *DDR_W_to){
  /* Variable Declaration */
  ap_uint<MAX_INP * PACK_NUM * W_BIT> packing_in;
  ap_uint<256> packing_in_256b;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
      FILE* fpw  = fopen("w_stream_pex1_gold.txt", "wb");
#endif
  int cnt=0;

    for(int rep=0;rep<2;rep++){
        for(int m=0;m<M/(MAX_OUP*2);m++){
            for(int n=0;n<N;n++){
                for(int y=0;y<MAX_OUP;y++){
                    for(int sm=0;sm<PACK_NUM;sm++){
                        packing_in=packing_in>>W_BIT;
                        packing_in(MAX_INP * PACK_NUM * W_BIT-1,(MAX_INP * PACK_NUM-1) * W_BIT)=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];


                        #ifdef GLODEN_DEBUG
                        if(y==1){
                            ap_int<W_BIT> w0=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];

                            fprintf(fpw, "%d\n", (int)w0);
                        }
                        #endif

                    }
                }
            packing_in_256b=((ap_uint<256 - MAX_INP * PACK_NUM * W_BIT>)0, packing_in);
            DDR_W_to[cnt]=packing_in;
            cnt++;
            }
        }
    }


#ifdef GLODEN_DEBUG
      fclose(fpw);
#endif

}


void host_DDR_W(ap_int<W_BIT> W_from[N][M], ap_uint<MAX_OUP * PACK_NUM * W_BIT>*DDR_W_to){
  /* Variable Declaration */
  ap_uint<MAX_INP * PACK_NUM * W_BIT> packing_in;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
      FILE* fpw  = fopen("w_stream_pex1_gold.txt", "wb");
#endif
  int cnt=0;

    for(int m=0;m<M/(MAX_OUP*2);m++){
        for(int n=0;n<N;n++){
            for(int y=0;y<MAX_OUP;y++){
                for(int sm=0;sm<PACK_NUM;sm++){
                    packing_in=packing_in>>W_BIT;
                    packing_in(MAX_INP * PACK_NUM * W_BIT-1,(MAX_INP * PACK_NUM-1) * W_BIT)=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];


                    #ifdef GLODEN_DEBUG
                    if(y==1){
                        ap_int<W_BIT> w0=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];

                        fprintf(fpw, "%d\n", (int)w0);
                    }
                    #endif

                }
            }
            DDR_W_to[cnt]=packing_in;
            cnt++;
        }
    }


#ifdef GLODEN_DEBUG
      fclose(fpw);
#endif

}


void host_DDR_W_256b(ap_int<W_BIT> W_from[N][M], ap_uint<256>*DDR_W_to){
  /* Variable Declaration */
  ap_uint<MAX_OUP * PACK_NUM * W_BIT> packing_in;
  ap_uint<256> packing_in_256b;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
      FILE* fpw  = fopen("w_stream_pex1_gold.txt", "wb");
#endif
  int cnt=0;

    for(int m=0;m<M/(MAX_OUP*2);m++){
        for(int n=0;n<N;n++){
            for(int y=0;y<MAX_OUP;y++){
                for(int sm=0;sm<PACK_NUM;sm++){
                    packing_in=packing_in>>W_BIT;
                    packing_in(MAX_OUP * PACK_NUM * W_BIT-1,(MAX_OUP * PACK_NUM-1) * W_BIT)=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];


                    #ifdef GLODEN_DEBUG
                    if(y==1){
                        ap_int<W_BIT> w0=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];

                        fprintf(fpw, "%d\n", (int)w0);
                    }
                    #endif

                }
            }
            packing_in_256b=((ap_uint<256 - MAX_OUP * PACK_NUM * W_BIT>)0, packing_in);
            DDR_W_to[cnt]=packing_in;
            cnt++;
        }
    }


#ifdef GLODEN_DEBUG
      fclose(fpw);
#endif

}


void host_DDR_W_128b(ap_int<W_BIT> W_from[N][M], ap_uint<128>*DDR_W_to){
  /* Variable Declaration */
  ap_uint<MAX_OUP * PACK_NUM * W_BIT> packing_in;
  ap_uint<128> packing_in_128b;
  /* Variable Declaration */

#ifdef GLODEN_DEBUG
      FILE* fpw  = fopen("w_stream_pex1_gold.txt", "wb");
#endif
  int cnt=0;

    for(int m=0;m<M/(MAX_OUP*2);m++){
        for(int n=0;n<N;n++){
            for(int y=0;y<MAX_OUP;y++){
                for(int sm=0;sm<PACK_NUM;sm++){
                    packing_in=packing_in>>W_BIT;
                    packing_in(MAX_OUP * PACK_NUM * W_BIT-1,(MAX_OUP * PACK_NUM-1) * W_BIT)=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];


                    #ifdef GLODEN_DEBUG
                    if(y==1){
                        ap_int<W_BIT> w0=W_from[n][m*MAX_OUP*PACK_NUM+y*PACK_NUM+sm];

                        fprintf(fpw, "%d\n", (int)w0);
                    }
                    #endif

                }
            }
            packing_in_128b=((ap_uint<128 - MAX_OUP * PACK_NUM * W_BIT>)0, packing_in);
            DDR_W_to[cnt]=packing_in;
            cnt++;
        }
    }


#ifdef GLODEN_DEBUG
      fclose(fpw);
#endif

}


void print_pe_out(float O_golden[R][M]){
    char fp_name[100];

    for(int x=0; x< MAX_INP; x++){
        for(int y=0; y< MAX_OUP; y++){
            sprintf(fp_name,"gloden_out_pe%d%d.txt",x,y);
            FILE* fp_true_pe00 = fopen(fp_name, "wb");
            cout<<"begine pe"<<x<<"--"<<y<<"......."<<endl;
            for(int i=0; i< R/(MAX_INP*2); i++){
                for(int j=0; j< M/(MAX_OUP*2); j++){
                    fprintf(fp_true_pe00, "%d\n", (int)O_golden[0+i*8+2*x][0+j*8+2*y]);
                    fprintf(fp_true_pe00, "%d\n", (int)O_golden[0+i*8+2*x][1+j*8+2*y]);
                    fprintf(fp_true_pe00, "%d\n", (int)O_golden[1+i*8+2*x][j*8+2*y]);
                    fprintf(fp_true_pe00, "%d\n", (int)O_golden[1+i*8+2*x][1+j*8+2*y]);
                    cout<<0+i*8+2*x<<"--"<<0+j*8+2*y<<endl;
                    cout<<0+i*8+2*x<<"--"<<1+j*8+2*y<<endl;
                    cout<<0+i*8+2*x<<"--"<<0+j*8+2*y<<endl;
                    cout<<0+i*8+2*x<<"--"<<1+j*8+2*y<<endl;
                    
                }
            }
            fclose(fp_true_pe00);

        }
    }


}


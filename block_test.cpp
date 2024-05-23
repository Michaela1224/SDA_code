// #include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
// #define __gmp_const const

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include "block_top.h"
#include "test.h"
#include "config_test.h"
// using namespace std;
#define GENERATE_BIN

/*
int main(void){

	// conv
	ap_uint<128> *conv3_ddr_a;

	conv3_ddr_a = (ap_uint<128>*)malloc((2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2))*sizeof(ap_uint<128>));  //typedef ap_int<32>  ADT4;


	FILE* fp_QAin = fopen("conv_in_oup.bin", "rb");

	fread(conv3_ddr_a,sizeof(ap_uint<128>),(2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2)),fp_QAin);
	// fread(conv3_ddr_shortcut,sizeof(ap_uint<256>),(CONV_R*CONV_C*CONV_N/(MAX_INP * PACK_NUM)),fp_QAin);

	fclose(fp_QAin);


	ap_uint<128> *conv3_ddr_w;

	conv3_ddr_w =(ap_uint<128>*)malloc(((CONV_K*CONV_N)/MAX_INP)*(CONV_M)*2*sizeof(ap_uint<128>));  //typedef ap_int<32>  ADT4;

	FILE* fp_QWin = fopen("conv3_w.bin", "rb");

	fread(conv3_ddr_w,sizeof(ap_uint<128>),2*((CONV_K*CONV_N)/MAX_INP)*CONV_M,fp_QWin);

	cout<<conv3_ddr_w[0]<<endl;

	fclose(fp_QWin);

	//  fc ddr-shortcut-parameter
	// ap_uint<128> *conv3_ddr_fc_shortcut;
	// conv3_ddr_fc_shortcut = (ap_uint<128>*)malloc((CONV_M/(MAX_OUP))*2*sizeof(ap_uint<128>));  //typedef ap_int<32>  ADT4;

	// FILE* fp_QFC = fopen("conv3_fcvu_short.bin", "rb");

	// fread(conv3_ddr_fc_shortcut,sizeof(ap_uint<128>),(CONV_M/(MAX_OUP))*2,fp_QFC);
		// fclose(fp_QFC);

	ap_uint<128> *conv3_ddr_shortcut;
	cout<<"Byte: "<<sizeof(ap_uint<MAX_INP * PACK_NUM * IN_BIT>)<<endl;
	conv3_ddr_shortcut = (ap_uint<128>*)malloc((CONV_R*CONV_C*CONV_M/(2 * MAX_OUP))*2*sizeof(ap_uint<128>));  // 8192

	FILE* fpa_shortcut = fopen("conv3_fm_short.bin", "wb");

	fwrite(conv3_ddr_shortcut,sizeof(ap_uint<128>),(CONV_R*CONV_C*CONV_M/(2 * MAX_OUP))*2,fpa_shortcut);

	fclose(fpa_shortcut);




	// bia parameter//////////////////

	// 这里 待修改

	unsigned bias_num=CONV_M/(MAX_OUP/2); 

	unsigned factor_num=1;

	const unsigned PACKING_MAX_NORM_PE_NUM=128/(MAX_NORM_PE*16*2);
	unsigned norm_gamma_beta_num=CONV_M/(PACKING_MAX_NORM_PE_NUM*MAX_NORM_PE); 
	const unsigned PACKING_MAX_NORM_PE_PTF_FACTOR_NUM=2; // attention: 当Y不为20要做修改  512/(2*)
	unsigned norm_ptf_num=CONV_M/(PACKING_MAX_NORM_PE_PTF_FACTOR_NUM*MAX_OUP); 
	
	unsigned total_num=bias_num+factor_num+norm_gamma_beta_num+norm_ptf_num;
	


	ap_uint<128> *input_ln_parameter = (ap_uint<128> *)malloc((total_num)*sizeof(ap_uint<128>));
	


	FILE* fp_QBin = fopen("conv3_bias.bin", "rb");

	fread(input_ln_parameter,sizeof(ap_uint<128>),total_num,fp_QBin);

	fclose(fp_QBin);





	unsigned layer_bias_offset=0;

	// unsigned layermode=0;




	ap_uint<128>* ddr_fm_back;
	ap_uint<128>* ddr_fm_shortcut_back;

	ddr_fm_back = (ap_uint<128>*)malloc((2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2))*sizeof(ap_uint<128>)); 
	ddr_fm_shortcut_back = (ap_uint<128>*)malloc((2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2))*sizeof(ap_uint<128>));




	unsigned which_path;
	which_path=2;

	bool CONV1_TO_MM_EN;

	CONV1_TO_MM_EN=true;

	do_compute_top(conv3_ddr_a, conv3_ddr_w ,input_ln_parameter,conv3_ddr_shortcut, ddr_fm_back, ddr_fm_shortcut_back, layer_bias_offset, CONV_R, CONV_C, CONV_N, CONV_M,CONV_D,
	                which_path,CONV1_TO_MM_EN);



	// #ifdef GENERATE_BIN
	// 	FILE* fpout_short = fopen("conv_out_short.bin", "wb");

	// 	fwrite(ddr_fm_shortcut_back,sizeof(ap_uint<128>),(2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2)),fpout_short);

	// 	fclose(fpout_short);
	// #endif


	#ifdef GENERATE_BIN
		FILE* fpout = fopen("conv_out.bin", "wb");

		fwrite(ddr_fm_back,sizeof(ap_uint<128>),(2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2)),fpout);

		fclose(fpout);
	#endif



	ap_uint<128> temp_128b;
	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;

	ap_uint<IN_BIT> temp_8b0;
	ap_uint<IN_BIT> temp_8b1;

	FILE* fp1 = fopen("result_verify_norm_1.txt", "wb");

	for(int i=0; i<2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2);i++){
#pragma HLS PIPELINE II=1
		temp_128b=ddr_fm_back[i];
		temp=temp_128b;
		for(int j=0; j<MAX_OUP/2;j++){
			(temp_8b1,temp_8b0)=temp((j+1)*IN_BIT * 2-1,j*IN_BIT * 2);
			fprintf(fp1, "%d\n", int(temp_8b0));
			fprintf(fp1, "%d\n", int(temp_8b1));
		}
	}

	fclose(fp1);

// 	FILE* fp2 = fopen("result_verify_shortcut_1.txt", "wb");

// 	for(int i=0; i<2*CONV_R*CONV_C*CONV_M/(MAX_OUP*2);i++){
// #pragma HLS PIPELINE II=1
// 		temp_128b=ddr_fm_shortcut_back[i];
// 		temp=temp_128b;
// 		for(int j=0; j<MAX_OUP/2;j++){
// 			(temp_8b1,temp_8b0)=temp((j+1)*IN_BIT * 2-1,j*IN_BIT * 2);
// 			fprintf(fp2, "%d\n", int(temp_8b0));
// 			fprintf(fp2, "%d\n", int(temp_8b1));
// 		}
// 	}

// 	fclose(fp2);





    free(conv3_ddr_a);
    free(conv3_ddr_w);
    // free(input_ln_parameter);
//    free(ddr_fm_output);



	return 0;
}
*/





int main(void){



	char fp_name_conv[100];
	ap_int<IN_BIT> A[R][N];
	ap_int<W_BIT> W[N][M];
	float O_golden[R][M];


	generate_mm_a(A);
	generate_mm_w(W);
	generate_mm_output(A,W,O_golden);



	ap_uint<128> *DDR_A;
	// cout<<"Byte: "<<sizeof(ap_uint<MAX_INP * PACK_NUM * IN_BIT>)<<endl;
	DDR_A = (ap_uint<128>*)malloc((R*M*2/(MAX_OUP*2))*sizeof(ap_uint<128>));  //typedef ap_int<32>  ADT4;

	// host_DDR_A_128b(A, DDR_A);
	// // cout <<"The Value of Var_p: \t" <<DDR_A[0]<< " \t Binary format: \t" <<DDR_A[0].to_string(16).c_str()<< '\n';
	// // cout<<DDR_A[1919]<<endl;

	FILE* fp_QAin = fopen("mm_fm_trans_160.bin", "rb");

	fread(DDR_A,sizeof(ap_uint<128>),(R*M*2/(MAX_OUP*2)),fp_QAin);
	// fread(conv3_ddr_shortcut,sizeof(ap_uint<256>),(CONV_R*CONV_C*CONV_N/(MAX_INP * PACK_NUM)),fp_QAin);

	fclose(fp_QAin);




	// ap_uint<128> *DDR_A_shortcut;

	// DDR_A_shortcut = (ap_uint<128>*)malloc((2*R*M/(MAX_OUP * PACK_NUM))*sizeof(ap_uint<128>));  //typedef ap_int<32>  ADT4;
	// generate_mm_shortcut_128btest(DDR_A_shortcut);

	// FILE* fp_QASHORT = fopen("mm_fm_trans_160.bin", "rb");

	// fread(DDR_A_shortcut,sizeof(ap_uint<128>),(R*M*2/(MAX_OUP*2)),fp_QASHORT);


	// fclose(fp_QASHORT);



	// #ifdef GENERATE_BIN
	// 	FILE* fpa_short = fopen("mm_fm_160_short.bin", "wb");

	// 	fwrite(DDR_A_shortcut,sizeof(ap_uint<128>),(2*R*M/(MAX_OUP * PACK_NUM)),fpa_short);

	// 	fclose(fpa_short);
	// #endif


	// ap_uint<128> *DDR_W;

	// DDR_W = (ap_uint<128>*)malloc((2*N*M/(MAX_OUP * PACK_NUM))*sizeof(ap_uint<128>));  //typedef ap_int<32>  ADT4;

	// host_DDR_W_Softmax_128b(W, DDR_W);

	// #ifdef GENERATE_BIN
	// 	FILE* fpw = fopen("mm_w_160.bin", "wb");

	// 	fwrite(DDR_W,sizeof(ap_uint<128>),(2*N*M/(MAX_OUP * PACK_NUM)),fpw);

	// 	fclose(fpw);
	// #endif

	ap_uint<128> *DDR_W;

	DDR_W = (ap_uint<128>*)malloc((N*M/(MAX_OUP * PACK_NUM))*sizeof(ap_uint<128>));  //typedef ap_int<32>  ADT4;

	host_DDR_W_128b(W, DDR_W);

	#ifdef GENERATE_BIN
		FILE* fpw = fopen("mm_w_160.bin", "wb");

		fwrite(DDR_W,sizeof(ap_uint<128>),(N*M/(MAX_OUP * PACK_NUM)),fpw);

		fclose(fpw);
	#endif



	unsigned bias_num=M/(MAX_OUP/2); 

	unsigned factor_num=1;

	// const unsigned PACKING_MAX_NORM_PE_NUM=128/(MAX_NORM_PE*16*2);
	// unsigned norm_gamma_beta_num=M/(PACKING_MAX_NORM_PE_NUM*MAX_NORM_PE); 
	// const unsigned PACKING_MAX_NORM_PE_PTF_FACTOR_NUM=2; // attention: 当Y不为20要做修改  512/(2*)
	// unsigned norm_ptf_num=M/(PACKING_MAX_NORM_PE_PTF_FACTOR_NUM*MAX_OUP); 
	
	// unsigned total_num=bias_num+factor_num+norm_gamma_beta_num+norm_ptf_num;
	
	unsigned total_num=bias_num+factor_num;

	ap_uint<128> *input_ln_parameter = (ap_uint<128> *)malloc((total_num)*sizeof(ap_uint<128>));
	

	generate_conv_allbias_128btest(input_ln_parameter,bias_num,total_num);


	#ifdef GENERATE_BIN
		FILE* fpb = fopen("mm_bias_160.bin", "wb");

		fwrite(input_ln_parameter,sizeof(ap_uint<128>),total_num,fpb);

		fclose(fpb);
	#endif





	ap_uint<128>* ddr_fm_back;
	ap_uint<128>* ddr_fm_shortcut_back;

	ddr_fm_back = (ap_uint<128>*)malloc((R*M*2/(MAX_OUP*2))*sizeof(ap_uint<128>)); 
	ddr_fm_shortcut_back = (ap_uint<128>*)malloc((R*M*2/(MAX_OUP*2))*sizeof(ap_uint<128>));




	unsigned which_path;
	which_path=10;

	bool CONV1_TO_MM_EN;

	CONV1_TO_MM_EN=false;
	
	do_compute_top(DDR_A, DDR_W,input_ln_parameter,0, ddr_fm_back, ddr_fm_shortcut_back, 0,0, R, 0, N, M,0,
	                which_path,CONV1_TO_MM_EN);



	#ifdef GENERATE_BIN
		FILE* fpout_short = fopen("mm_out_short_160.bin", "wb");

		fwrite(ddr_fm_shortcut_back,sizeof(ap_uint<128>),R*M*2/(MAX_OUP*2),fpout_short);

		fclose(fpout_short);
	#endif


	#ifdef GENERATE_BIN
		FILE* fpout = fopen("mm_out_160.bin", "wb");

		fwrite(ddr_fm_back,sizeof(ap_uint<128>),R*M*2/(MAX_OUP*2),fpout);

		fclose(fpout);
	#endif


	ap_uint<128> temp_128b;
	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;

	ap_uint<IN_BIT> temp_8b0;
	ap_uint<IN_BIT> temp_8b1;

	FILE* fp1 = fopen("result_verify_norm_1.txt", "wb");

	for(int i=0; i<R*M*2/(MAX_OUP*2);i++){
#pragma HLS PIPELINE II=1
		temp_128b=ddr_fm_back[i];
		temp=temp_128b;
		for(int j=0; j<MAX_OUP/2;j++){
			(temp_8b1,temp_8b0)=temp((j+1)*IN_BIT * 2-1,j*IN_BIT * 2);
			fprintf(fp1, "%d\n", int(temp_8b0));
			fprintf(fp1, "%d\n", int(temp_8b1));
		}
	}

	fclose(fp1);

	FILE* fp2 = fopen("result_verify_shortcut_1.txt", "wb");

	for(int i=0; i<R*M*2/(MAX_OUP*2);i++){
#pragma HLS PIPELINE II=1
		temp_128b=ddr_fm_shortcut_back[i];
		temp=temp_128b;
		for(int j=0; j<MAX_OUP/2;j++){
			(temp_8b1,temp_8b0)=temp((j+1)*IN_BIT * 2-1,j*IN_BIT * 2);
			fprintf(fp2, "%d\n", int(temp_8b0));
			fprintf(fp2, "%d\n", int(temp_8b1));
		}
	}

	fclose(fp2);



    free(DDR_A);
    free(DDR_W);
    // free(conv3_ddr_bias);
//    free(ddr_fm_output);



	return 0;
}

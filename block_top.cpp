#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include"diffusion-lib.h"
#include "config.h"
#include "param_sa.h"

//#define PRINT_DEBUG
//#define NROM_IN_DEBUG
//#define TEST_DEBUG
// #define INPUT_DEBUG
// #define MM_OS_RED_DEBUG
// #define RESULT_S2_DEBUG
// #define RESULT_DEBUG



void WriteMMFMParam(ap_uint<128>* in, ap_uint< MAX_INP * IN_BIT * PACK_NUM> mm_a_buf[MAX_MM_FM_LENGTH],
	const unsigned MM_N, const unsigned WhichPath){
#pragma HLS INLINE OFF
	if(WhichPath>=5){

		WriteMMFMParam_MMTRANSFER(in,mm_a_buf,MM_N);
	}
	else{
		WriteMMFMParam_DIRECT(in,mm_a_buf,MM_N);
	}
	
}


void MMFMParam_Stream(ap_uint< MAX_INP * IN_BIT * PACK_NUM> mm_a_buf[MAX_MM_FM_LENGTH],
	stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> >& fifo_out_mm,
	const unsigned MM_PE_NUM,
	const unsigned MM_N,
	bool tran_en){
#pragma HLS INLINE OFF
	if (!tran_en) return;

	ap_uint< MAX_INP * IN_BIT * PACK_NUM > temp;
	for(unsigned j=0; j<MM_PE_NUM;j++){
		for(unsigned i=0; i<MM_N;i++){
		#pragma HLS PIPELINE II=1
			temp = mm_a_buf[i];
			// cout<<temp<<endl;
			fifo_out_mm.write(temp);
		}
	}
}





void ExtractPixels_AXI_MMA(
	ap_uint<128>* in,
	stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> >& out_mm,
	stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> >& out_conv,
	const unsigned MM_N,
	const unsigned MM_SIMD_NUM,
	const unsigned MM_PE_NUM,
    const unsigned OUT_W,
    const unsigned D,
    const unsigned OUT_H,
	const unsigned NumLines,
	const unsigned conv3_group,
	const bool SA_MODE,
	const unsigned WhichPath){

#ifdef INPUT_DEBUG
    //   FILE* fpa = fopen("a_stream_pe00_gold.txt", "wb");
    FILE* fpa = fopen("a_stream.txt", "wb");
#endif

	
	int cnt=0;
	int penum_cnt=0;
	int simdnum_cnt=0;
	unsigned addra;

	if(SA_MODE==false){

	bool arb = 0;
	bool trans_en = 0;

		for (unsigned rep = 0; rep < MM_SIMD_NUM; rep++) {
			if(arb==0){
				WriteMMFMParam_MMTRANSFER(in+rep*MM_N*4, mm_a_buffer0,MM_N);
				MMFMParam_Stream(mm_a_buffer1,out_mm,MM_PE_NUM, MM_N,trans_en);
			}
			else{
				WriteMMFMParam_MMTRANSFER(in+rep*MM_N*4, mm_a_buffer1,MM_N);
				MMFMParam_Stream(mm_a_buffer0,out_mm,MM_PE_NUM, MM_N,trans_en);
			}
			trans_en = 1;
			arb = !arb;
		}
		if(arb==0){
			MMFMParam_Stream(mm_a_buffer1,out_mm,MM_PE_NUM, MM_N,trans_en);
		}
		else{
			MMFMParam_Stream(mm_a_buffer0,out_mm,MM_PE_NUM, MM_N,trans_en);
		}
	}
	else{

//		if(WhichPath==0){
//			ExtractPixels_AXI_CONV_DIRECT(in,out_conv,NumLines,conv3_group);
//		}
//		else if(WhichPath==1||WhichPath==2){
			ExtractPixels_AXI_CONV_OUT_TO_IN(in,out_conv, NumLines,conv3_group);
//		}

	}

}






void CONV3BiasParam_Stream(ap_int<BIAS_BIT> conv3_bias[MAX_OUP][MAX_CONV3_BIAS_LENGTH],
	stream<ap_int<BIAS_BIT> >  fifo_bias_in[MAX_OUP],
	const unsigned PENUM,
	const unsigned OUT_H,
    const unsigned OUT_W,
	bool tran_en){
#pragma HLS INLINE OFF
	if (!tran_en) return;

	ap_int<BIAS_BIT> temp;

    
    for(unsigned j=0; j<OUT_H;j++){
		for(unsigned i=0; i<PENUM;i++){
			for(unsigned k=0; k<OUT_W/2;k++){
        #pragma HLS PIPELINE II=1
				for(unsigned m=0;m<MAX_OUP;m++){
					temp = conv3_bias[m][i];
					// cout<<temp<<endl;
					fifo_bias_in[m].write(temp);
				}
			}
        }
    }

}

void ExtractPixels_AXI_AllBias(
	ap_uint<128>* bias_in,
	const unsigned M,
	const unsigned D,
	const unsigned layer_offset,
	const bool COMPUTE_SA_MODE,
	const bool COMPUTE_NORM_MODE
    ){
#pragma HLS INLINE OFF



	ap_uint<16> temp_data;
	LN_BIAS_DB temp_fix_data;
	

	// SA BIAS加载
	unsigned sa_bias_offset=layer_offset;

	// attention: 
	unsigned bias_num=M/(MAX_OUP/2); 

    unsigned int bitIdx=0;
    unsigned int colIdx=0;
    unsigned int depthIdx=0;


	ap_uint<128> temp_axi_data_bias[2];
	ap_uint<256> temp_data_256b;
	ap_uint<MAX_OUP*16> temp_data_160b;

    for(unsigned i=0; i<bias_num;i++){
#pragma HLS PIPELINE II=1

        temp_axi_data_bias[bitIdx]=bias_in[sa_bias_offset+i];
		
		// cout <<"The Value of Var_p: \t" <<temp_axi_data<< " \t Binary format: \t" <<temp_axi_data.to_string(2).c_str()<< '\n';

		if(bitIdx==2-1){
			temp_data_256b=(temp_axi_data_bias[1],temp_axi_data_bias[0]);
			temp_data_160b=temp_data_256b;

			for(int c=0; c<MAX_OUP; c++){
			#pragma HLS UNROLL
				conv3_mm_bias_buffer[c][depthIdx] = temp_data_160b(BIAS_BIT*(c+1)-1, BIAS_BIT*c);
				
				// cout<<"c: "<<c<<" index:"<<depthIdx<<endl;

			}

		}

        if(bitIdx==2-1){
            bitIdx=0;

			if(depthIdx==MAX_CONV3_BIAS_LENGTH-1){
				depthIdx=0;
			}
			else{
				depthIdx++;
			}

		}
		else{
			bitIdx++;
		}

    }


	// scale_factor BIAS加载
	unsigned scale_factor_offset=sa_bias_offset+bias_num;

	scale_factor_buffer= bias_in[scale_factor_offset];

	// test 
	// ap_int<BIAS_BIT>  Layer_Scale=scale_factor_buffer(15,0);
	// cout<<Layer_Scale<<endl;

	// scale_factor_buffer[0]=temp_scale_factor(15,0);
	// scale_factor_buffer[1]=temp_scale_factor(31,16);
	// scale_factor_buffer[2]=temp_scale_factor(47,32);
	// scale_factor_buffer[3]=temp_scale_factor(63,48);
	// scale_factor_buffer[4]=temp_scale_factor(79,64);
	// scale_factor_buffer[5]=temp_scale_factor(95,80);
	
	unsigned pwf_factor_offset;
	unsigned norm_ptf_num;

    if(COMPUTE_NORM_MODE==true){
		// scale_factor BIAS加载
		unsigned gama_offset=scale_factor_offset+1;


		unsigned int depthIdx_p=0;
		unsigned int depthIdx_g=0;
		ap_uint<MAX_NORM_PE*16> temp_data_g0;
		ap_uint<16> temp_data0, temp_data1;
		LN_BIAS_DB temp_fix_data0,temp_fix_data1;
		const unsigned PACKING_MAX_NORM_PE_NUM=1;
		unsigned norm_gamma_beta_num=M/(PACKING_MAX_NORM_PE_NUM*MAX_NORM_PE); 

		ap_uint<128> temp_axi_data_gabe;

		for(int i=0; i<norm_gamma_beta_num;i++){  // 512b可以放
			temp_axi_data_gabe=bias_in[gama_offset+i];
			for(int j=0; j<PACKING_MAX_NORM_PE_NUM;j++){
			#pragma HLS PIPELINE II=1
				temp_data_g0=temp_axi_data_gabe((j+1)*MAX_NORM_PE*16-1,j*MAX_NORM_PE*16);
				
				// cout<< "temp_data_g0: "<<temp_data_g0<<endl;
				// cout<< "temp_data_g1: "<<temp_data_g1<<endl;

				for(int k=0; k<MAX_NORM_PE;k++){
					temp_data0=temp_data_g0((k+1)*16-1,k*16);
					temp_fix_data0.range(15,0)=temp_data0(15,0);

					
					ln_gamma_buffer[k][i*PACKING_MAX_NORM_PE_NUM+j]=temp_fix_data0;

					// cout<<"r: "<<k<<" c:"<< i*PACKING_MAX_NORM_PE_NUM+j <<"\ttemp_data_g0: "<<temp_data_g0<<endl;
					// cout<< "\ttemp_data_g1: "<<temp_data_g1<<endl;

					ln_beta_buffer[k][i*PACKING_MAX_NORM_PE_NUM+j]=temp_fix_data0;
					// cout<<"gamma input:"<<ln_gamma_buffer[cnt_gamma]<<endl;	
				}

			}

		}


		ap_uint<128> temp_axi_data_ptf;
		pwf_factor_offset=gama_offset+norm_gamma_beta_num;

		unsigned int numIdx0=0;
		unsigned int colIdx0=0;
		unsigned int depthIdx0=0;
		ap_uint<MAX_OUP*LN_PWF_FACTOR_BIT> temp_factor_wid;
		ap_uint<LN_PWF_FACTOR_BIT> temp_factor;

		const unsigned PACKING_MAX_NORM_PE_PTF_FACTOR_NUM=2; // attention: 当Y不为20要做修改  512/(2*)
		norm_ptf_num=M/(PACKING_MAX_NORM_PE_PTF_FACTOR_NUM*MAX_OUP); 
		
		for(int i=0; i<norm_ptf_num;i++){
				temp_axi_data_ptf=bias_in[pwf_factor_offset+i];
				for(int j=0; j<PACKING_MAX_NORM_PE_PTF_FACTOR_NUM;j++){
				#pragma HLS PIPELINE II=1
					temp_factor_wid=temp_axi_data_ptf((j+1)*(MAX_OUP*LN_PWF_FACTOR_BIT)-1,j*MAX_OUP*LN_PWF_FACTOR_BIT);
					for(int k=0;k<MAX_OUP;k++){
						temp_factor=temp_factor_wid((k+1)*2-1,k*2);
						if(COMPUTE_SA_MODE==false){
							ln_ptf_factor_buffer0[k][i*PACKING_MAX_NORM_PE_PTF_FACTOR_NUM+j]=temp_factor;
							ln_ptf_factor_buffer1[k][i*PACKING_MAX_NORM_PE_PTF_FACTOR_NUM+j]=temp_factor;
						}
						else if(COMPUTE_SA_MODE==true){
							if(numIdx0%2==0){
								ln_ptf_factor_buffer0[k][(numIdx0/2)*(D/MAX_OUP)+depthIdx0] =temp_factor;

//								cout<<"buf-A "<<"r: "<<k<<" c:"<<((numIdx0/2)*(D/MAX_OUP)+depthIdx0)<<endl;
							}
							else{
								ln_ptf_factor_buffer1[k][(numIdx0/2)*(D/MAX_OUP)+depthIdx0] =temp_factor;

//								cout<<"buf-B "<<"r: "<<k<<" c:"<<((numIdx0/2)*(D/MAX_OUP)+depthIdx0)<<endl;
							}
						}
					}
						if(depthIdx0==D/MAX_OUP-1){
							depthIdx0=0;
							if(numIdx0==M/D-1){
								numIdx0=0;
							}
							else{
								numIdx0++;
							}
						}
						else{
							depthIdx0++;
						}

				}
			}
	} 




}



void ExtractPixels_AXI_MMW_CONV(
	ap_uint< 128>* in,
	stream<ap_uint<MAX_OUP * W_BIT * PACK_NUM> >& out,
	stream<ap_uint<MAX_INP * CONV_K *W_BIT> > fifo_W_in[MAX_A_COL],
	const unsigned MM_N,
	const unsigned MM_SIMD_NUM,
	const unsigned MM_PE_NUM,
	const unsigned NumLines,
	const unsigned OUT_H,
	const unsigned GROUPS,
	const bool SA_MODE,
	const unsigned layer_weight_offset )
{

    if(SA_MODE==false){ // MM

		ap_uint< 128 > temp_128b;
		ap_uint< MAX_OUP * W_BIT * PACK_NUM > temp;

		for (unsigned rep = 0; rep < MM_SIMD_NUM; rep++) {
	#pragma HLS loop_tripcount min=NumLines max=NumLines
			for(unsigned i=0; i<MM_N*MM_PE_NUM;i++){
	#pragma HLS PIPELINE II=1
				temp_128b = in[layer_weight_offset+i];
				temp=temp_128b;
				out.write(temp);
			}
		}
	}
	else if(SA_MODE==true){
		unsigned conv3_weight_num=(MAX_A_COL)*NumLines;

		bool arb = 0;
		bool trans_en = 0;

		for (unsigned rep = 0; rep < GROUPS; rep++) {
			if(arb==0){
				WriteConv3WeightParam(in+layer_weight_offset+rep*conv3_weight_num,conv3_w_buffer0,conv3_weight_num);
				CONV3WeightParam_Stream(conv3_w_buffer1,fifo_W_in,NumLines,OUT_H,trans_en);
			}
			else{
				WriteConv3WeightParam(in+layer_weight_offset+rep*conv3_weight_num,conv3_w_buffer1,conv3_weight_num);
				CONV3WeightParam_Stream(conv3_w_buffer0,fifo_W_in,NumLines,OUT_H,trans_en);
			}
			trans_en = 1;
			arb = !arb;
		}
		if(arb==0){
			CONV3WeightParam_Stream(conv3_w_buffer1,fifo_W_in,NumLines,OUT_H,trans_en);
		}
		else{
			CONV3WeightParam_Stream(conv3_w_buffer0,fifo_W_in,NumLines,OUT_H,trans_en);
		}


	}
}















void DemuxStream2_ARRAY(
	stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > in[MAX_A_ROW][MAX_A_COL][SA_OUP], 
	stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > out1[MAX_A_ROW][MAX_A_COL][SA_OUP], 
	stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > out2[MAX_A_ROW][MAX_A_COL][SA_OUP], 
    const unsigned NumLines,
	const bool mode)
{
	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_A_COL; c++){
        for(unsigned int m = 0; m < SA_OUP; m++){
          for(unsigned int r = 0; r < MAX_A_ROW; r++){	
			ap_uint<ACC_BIT * PACK_OUT_NUM> temp = in[r][c][m].read();
			if (mode == false)
				out1[r][c][m].write(temp);  // to_mm
			else if (mode==true)
				out2[r][c][m].write(temp);   // to_conv3
		  }
		}
	  }
	}
}



/*
void DemuxStreamN_OUP(
	stream<ap_uint<DEQUAN_BIT * 2> > in[SA_OUP], 
	stream<ap_uint<DEQUAN_BIT * 2> > out1[SA_OUP], 
	stream<ap_uint<DEQUAN_BIT * 2> > out2[SA_OUP], 
	const unsigned block_mode, 
    const unsigned NumLines){

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < SA_OUP; c++){
		ap_uint<DEQUAN_BIT * 2> temp = in[c].read();

		if(block_mode==0){
			out1[c].write(temp);  // shortcut quan 
		}
		else if(block_mode==1){
			out2[c].write(temp);  // shortcut quan 

		}


	  }
	}

}
*/


void DuplicateStreamN_OUP(
	stream<ap_uint<DEQUAN_BIT * 2> > in[MAX_OUP], 
	stream<ap_uint<DEQUAN_BIT * 2> > out1[MAX_OUP], 
	stream<ap_uint<DEQUAN_BIT * 2> > out2[MAX_OUP],
    const unsigned NumLines,
	const bool NORM_MODE,
	const bool SHORCUT_QUAN_MODE,
	const bool SOFTMAX_MODE,
	const bool EBMULT_MODE,
	const bool GELU_MODE,
	const bool TRANSPOSE_MODE,
	const bool SHORTCUT_ADD_MODE
	){
	
	if((NORM_MODE==false) && (SOFTMAX_MODE==false)&& (GELU_MODE==false) && (TRANSPOSE_MODE==false)
	    && (SHORCUT_QUAN_MODE==false)&&(EBMULT_MODE==false)){
		return;
	}

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_OUP; c++){
		ap_uint<DEQUAN_BIT * 2> temp = in[c].read();

		if(SHORCUT_QUAN_MODE&&SHORTCUT_ADD_MODE==false&&EBMULT_MODE==false){
			out1[c].write(temp);  // shortcut quan 
		}
		if(NORM_MODE||SOFTMAX_MODE||GELU_MODE||TRANSPOSE_MODE||EBMULT_MODE){
			out2[c].write(temp);  // norm 
		}

	  }
	}

}




void Dequan_to_Res( stream<ap_uint<ACC_BIT*2> > fifo_C_in[MAX_OUP],
                stream<ap_uint<DEQUAN_BIT*2> > fifo_C_out[MAX_OUP],
                // const unsigned INCOFFSET,
                const unsigned PENUM,
				const unsigned SIMDNUM,
                const unsigned OUT_W,
                const unsigned OUT_H,
                const unsigned GROUPS,
                const unsigned Numlines,
				const bool 	SA_MODE) {


unsigned int loop0,loop1,loop2,loop3;



ap_int<ACC_BIT> data0, data1;
ap_int<BIAS_BIT> bias_temp;


ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> res_out0,res_out1;
ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> fc_temp;
ap_uint<DEQUAN_BIT> out0, out1;

    unsigned int outfoldIdx=0;
    unsigned int w=0;
    unsigned int h=0;
    unsigned int g=0;
	unsigned int index=0;
	
    ap_int<BIAS_BIT>  Layer_Scale=scale_factor_buffer(15,0);

//	 cout<<"Layer_Scale: "<<Layer_Scale<<endl;

	if(SA_MODE==false){
		loop0=2;
		loop1=MAX_INP;
		loop2=PENUM;
		loop3=SIMDNUM;
	}
	else if(SA_MODE==true){
		loop0=OUT_W/2;
		loop1=PENUM;
		loop2=OUT_H;
		loop3=GROUPS;		
	}

  for (unsigned int rep = 0; rep < Numlines; rep++) { // 40
#pragma HLS PIPELINE II=1

		if(SA_MODE==false){
			index=w+h*loop0;
		}
		else if(SA_MODE==true){
			index=outfoldIdx+g*loop1;
		}

      for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL
        (data1, data0) = fifo_C_in[i].read();

        bias_temp = conv3_mm_bias_buffer[i][index];

//        if(i==0){
//          cout<<INCOFFSET+outfoldIdx+g*PENUM<<"value: "<<bias_temp<<endl;
//        }


        out0=DeQuan_Bias_Unit<ACC_BIT, DEQUAN_BIT,DEQUAN_INTEGER_BIT, BIAS_BIT, Shift_Num>(data0, bias_temp,Layer_Scale);
        out1=DeQuan_Bias_Unit<ACC_BIT, DEQUAN_BIT,DEQUAN_INTEGER_BIT, BIAS_BIT, Shift_Num>(data1, bias_temp,Layer_Scale);

		// out0=res_out0(DEQUAN_BIT-1,0);
		// out1=res_out1(DEQUAN_BIT-1,0);

        fifo_C_out[i].write((out1, out0));
      }

      if(w==loop0-1){
          w=0;
          if(outfoldIdx==loop1-1){
              outfoldIdx=0;
              if(h==loop2-1){
                h=0;
                if(g==loop3-1){
                  g=0;
                }
                else{
                  g++;
                }
              }
              else{
                h++;
              }
          }
          else{
              outfoldIdx++;
          }
      }
      else{
          w++;
      }

  }




}





void WriteLNParam(stream<ap_uint<DEQUAN_BIT*2> > in[MAX_OUP], 
	ap_uint<MAX_OUP*ILN_WIDTH*2> SHORTCUT_buffer[MAX_SHORTCUT_NORM_INBUF_LENGTH], 
	ap_uint<ILN_WIDTH*2*2> ln_buf[MAX_OUP][MAX_NORM_INBUF_LENGTH],
	stream<ap_uint<DEQUAN_BIT*2> > no_norm_out[MAX_OUP], 
	ap_uint<LN_PWF_FACTOR_BIT> buf_ln_ptf_factor[MAX_OUP][MAX_NORM_PWF_FACTOR_LENGTH],
	De_Quan_DB shortcut_deq_factor,
	ap_uint<ILN_N_MEAN_WIDTH> mean_unnorm[MAX_INP][2],
	ap_uint<ILN_N_VAR_WIDTH> var_unnorm[MAX_INP][2],
	const unsigned PENUM,
	const unsigned OUT_W,
	const unsigned OUT_H,
	const unsigned group_num,
	const bool SA_Mode,
	const bool SHORTCUT_MODE,
	const bool FCU_MODE
	){
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=buf_ln_ptf_factor dim=1 complete
//#pragma HLS BIND_STORAGE variable=ln_ptf_factor_buffer type=ram_2p impl=bram





#ifdef NROM_IN_DEBUG
    FILE* fp_parameter= fopen("norm_ptf.txt", "wb");

#endif


	ap_uint< DEQUAN_BIT*2 > temp;

	ap_uint<DEQUAN_BIT> dequan_x0, dequan_x1;
	De_Quan_DB dequan_fixp_x0, dequan_fixp_x1;

	De_Quan_DB no_norm_fixp_x0, no_norm_fixp_x1;
	ap_uint<ILN_WIDTH*2> norm_x0_16bit, norm_x1_16bit;

	ap_uint<ILN_WIDTH> temp_x0, temp_x1;
	ap_uint<ILN_WIDTH*2> temp_x0_16bit, temp_x1_16bit;
	ap_uint<ILN_WIDTH> x0_4b, x1_4b;

	ap_uint<2> shift_factor_mean=0;
	ap_uint<3> shift_factor_var;

	ap_uint<ILN_N_MEAN_WIDTH> mean_unnorm_temp[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=mean_unnorm_temp complete dim=2
	ap_uint<ILN_N_VAR_WIDTH> var_unnorm_temp[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=var_unnorm_temp complete dim=2

	ap_uint<ILN_N_MEAN_WIDTH> mean=0;
	ap_uint<ILN_N_VAR_WIDTH> var=0;

	ap_uint<12> inter_var0,inter_var1;
	unsigned int outdIdx=0;
    unsigned int w=0;
    unsigned int h=0;

	unsigned int loop0,loop1,loop2;
	unsigned int numlines;
	// unsigned int index_ini=group_num*(D/MAX_OUP);

	ap_uint<MAX_OUP*ILN_WIDTH*2> shortcut_oup_temp;
	ap_uint<2*ILN_WIDTH> shortcut_temp;
	ap_int<ILN_WIDTH> shortcut_value0;
	ap_int<ILN_WIDTH> shortcut_value1;

	De_Quan_DB dequan_fixp_shortcut0[MAX_OUP], dequan_fixp_shortcut1[MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=dequan_fixp_shortcut0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=dequan_fixp_shortcut1 complete dim=0
	unsigned int index_ini;
	unsigned int index_input,index_factor;

	unsigned int buf_in_index;

	if(SA_Mode==false){  // MM
		loop0=2;
		loop1=MAX_INP;
		loop2=PENUM; // M/(MAX_OUP*2)
		index_ini=0;
		numlines= PENUM*MAX_INP*2;
	}
	else if(SA_Mode==true){
		loop0=OUT_W/2;
		loop1=PENUM;  // D/MAX_OUP
		loop2=OUT_H;
		index_ini=(group_num/2)*PENUM;
		numlines= OUT_H*(PENUM)*(OUT_W/2);
	}


//	cout<< index_ini<<endl;

	for(unsigned i=0; i<MAX_INP;i++){
// #pragma HLS UNROLL
		mean_unnorm_temp[i][0]=0;
		mean_unnorm_temp[i][1]=0;
		var_unnorm_temp[i][0]=0;
		var_unnorm_temp[i][1]=0;
	}


	for(unsigned m=0; m<numlines;m++){
#pragma HLS PIPELINE II=1

		if(SA_Mode==false){
			index_factor=loop0*h+w;
		}
		else if(SA_Mode==true){
			index_factor=index_ini+outdIdx;
		}

		if(SHORTCUT_MODE&&FCU_MODE==false){
			shortcut_oup_temp=SHORTCUT_buffer[m];
		}
		else if (SHORTCUT_MODE&&FCU_MODE==true){
			shortcut_oup_temp=SHORTCUT_buffer[outdIdx];
		}
		else{
			shortcut_oup_temp=0;
		}

		for(unsigned j=0; j<MAX_OUP;j++){
			// shortcut_add
			shortcut_temp=shortcut_oup_temp((j+1)*2*ILN_WIDTH-1,j*2*ILN_WIDTH);

			if(SHORTCUT_MODE){
				(shortcut_value1,shortcut_value0)=shortcut_temp;
				dequan_fixp_shortcut0[j]=shortcut_value0*shortcut_deq_factor;
				dequan_fixp_shortcut1[j]=shortcut_value1*shortcut_deq_factor;
			}
			else if (FCU_MODE){
				dequan_fixp_shortcut0[j](ILN_WIDTH*2-1,0)=shortcut_temp(ILN_WIDTH*2-1,0);
				dequan_fixp_shortcut1[j](ILN_WIDTH*2-1,0)=shortcut_temp(ILN_WIDTH*2-1,0);
			}
			else{
				dequan_fixp_shortcut0[j]=0;
				dequan_fixp_shortcut1[j]=0;
			}
		}


		for(unsigned i=0; i<MAX_OUP;i++){



//			cout<<temp<<endl;

			temp = in[i].read();
			
//			cout<<"endl....."<<endl;
			

			(dequan_x1,dequan_x0)=temp;

			#ifdef LNORM_DEBUG
				cout<<"dequan_x0:"<<dequan_x0<<endl;
				cout<<"dequan_x1:"<<dequan_x1<<endl;
			#endif

			dequan_fixp_x0(DEQUAN_BIT-1,0)=dequan_x0(DEQUAN_BIT-1,0);
			dequan_fixp_x1(DEQUAN_BIT-1,0)=dequan_x1(DEQUAN_BIT-1,0);

			#ifdef RESULT_DEBUG
				cout<<"temp_x0:"<<temp_x0<<endl;
				cout<<"temp_x1:"<<temp_x1<<endl;
			#endif
			// 均值累加 
				shift_factor_mean=buf_ln_ptf_factor[i][index_factor];

			#ifdef NROM_IN_DEBUG
				ap_uint<LN_PWF_FACTOR_BIT> test;
				test=buf_ln_ptf_factor[i][index_factor];
				fprintf(fp_parameter, "%lf\n", double(test));
			#endif


			#ifdef RESULT_DEBUG
				cout<<"shift_factor_mean:"<<shift_factor_mean<<endl;
			#endif

			#ifdef RESULT_DEBUG
				cout<<"dequan_fixp_x0:"<<dequan_fixp_x0<<endl;
				cout<<"dequan_fixp_x1:"<<dequan_fixp_x1<<endl;
				cout<<"shift_factor_mean:"<<shift_factor_mean<<endl;
			#endif

			// 存入BRAM
			no_norm_fixp_x0=dequan_fixp_x0+dequan_fixp_shortcut0[i];

			no_norm_fixp_x1=dequan_fixp_x1+dequan_fixp_shortcut1[i];

			temp_x0=ap_uint<ILN_WIDTH>((no_norm_fixp_x0)>>shift_factor_mean);
			
			temp_x1=ap_uint<ILN_WIDTH>((no_norm_fixp_x1)>>shift_factor_mean);

			#ifdef RESULT_DEBUG
				cout<<"temp_x0:"<<temp_x0<<endl;
				cout<<"temp_x1:"<<temp_x1<<endl;
			#endif
			temp_x0_16bit=temp_x0;
			temp_x1_16bit=temp_x1;

			norm_x0_16bit(DEQUAN_BIT-1,0)=no_norm_fixp_x0(DEQUAN_BIT-1,0);
			norm_x1_16bit(DEQUAN_BIT-1,0)=no_norm_fixp_x1(DEQUAN_BIT-1,0);

			if(SHORTCUT_MODE&&FCU_MODE==false){
				no_norm_out[i].write((norm_x1_16bit,norm_x0_16bit));
			}

			ln_buf[i][h*loop1*loop0+outdIdx*loop0+w]=(temp_x1_16bit,temp_x0_16bit);






			#ifdef RESULT_DEBUG
				cout<<"mean:"<<mean_paralel[i]<<endl;
			#endif

			mean_unnorm_temp[outdIdx][0]=mean_unnorm_temp[outdIdx][0]+((ap_uint<12>(temp_x0))<<shift_factor_mean);
			mean_unnorm_temp[outdIdx][1]=mean_unnorm_temp[outdIdx][1]+((ap_uint<12>(temp_x1))<<shift_factor_mean);


			#ifdef RESULT_DEBUG
				cout<<"mean_s:"<<((ap_uint<20>(temp_x0+temp_x1))<<shift_factor_mean)<<endl;
				cout<<"mean:"<<mean_paralel[i]<<endl;
			#endif
			// var累加
			shift_factor_var=((ap_uint<3>)shift_factor_mean)<<1;

			#ifdef RESULT_DEBUG
				cout<<"shift_factor_var:"<<shift_factor_var<<endl;
			#endif

			inter_var0=compute_mean_var<ILN_WIDTH>(temp_x0);

			#ifdef RESULT_DEBUG
				cout<<"inter_var0:"<<inter_var0<<endl;
			#endif
			
			inter_var1=compute_mean_var<ILN_WIDTH>(temp_x1);

			#ifdef RESULT_DEBUG
				cout<<"inter_var1:"<<inter_var1<<endl;
			#endif

			var_unnorm_temp[outdIdx][0]=var_unnorm_temp[outdIdx][0]+((ap_uint<20>(inter_var0))<<shift_factor_var);
			var_unnorm_temp[outdIdx][1]=var_unnorm_temp[outdIdx][1]+((ap_uint<20>(inter_var1))<<shift_factor_var);
			// var_parallel[i]=var_parallel[i]+((ap_uint<21>(inter_var0+inter_var1))<<shift_factor_var);

			#ifdef RESULT_DEBUG
				cout<<"var:"<<var_parallel[i]<<endl;
			#endif

		}



		// #ifdef PRINT_DEBUG
		// 	cout<<"mean_unnorm:"<<mean<<endl;
		// 	cout<<"var_unnorm:"<<var<<endl;
		// #endif

		if(w==loop0-1){
			w=0;
			if(outdIdx==loop1-1){
				outdIdx=0;
				if(h==loop2-1){
					h=0;
				}
				else{
					h++;
				}
			}
			else{
				outdIdx++;
			}
		}
		else{
			w++;
		}


	}


	for(unsigned i=0; i<loop1;i++){
		mean_unnorm[i][0]=mean_unnorm_temp[i][0];
		mean_unnorm[i][1]=mean_unnorm_temp[i][1];
		var_unnorm[i][0]=var_unnorm_temp[i][0];
		var_unnorm[i][1]=var_unnorm_temp[i][1];

		#ifdef NROM_IN_DEBUG
			cout<<"mean_unnorm[i][0]:"<<mean_unnorm[i][0]<<endl;
			cout<<"mean_unnorm[i][1]:"<<mean_unnorm[i][1]<<endl;
			cout<<"var_unnorm[i][0]:"<<var_unnorm[i][0]<<endl;
			cout<<"var_unnorm[i][1]:"<<var_unnorm[i][1]<<endl;
		#endif


	}


	#ifdef RESULT_DEBUG
		cout<<"mean_unnorm:"<<mean<<endl;
		cout<<"var_unnorm:"<<var<<endl;
	#endif


#ifdef NROM_IN_DEBUG

	fclose(fp_parameter);

#endif

}



void compute_mean_var(ap_uint<ILN_N_MEAN_WIDTH> mean_unnorm,ap_uint<ILN_N_VAR_WIDTH> var_unnorm,
	ap_uint<17> ln_rcd_factor,
	ap_ufixed<ILN_MEAN_WIDTH+8, ILN_MEAN_WIDTH> &mean_norm,
	ap_ufixed<18, 2> &std_inv ){

	// ap_ufixed<ILN_MEAN_WIDTH+8, ILN_MEAN_WIDTH> mean_norm;
	ap_ufixed<ILN_VAR_WIDTH+8, ILN_VAR_WIDTH> var_norm;
	// ap_ufixed<18, 2> std_inv;


	#ifdef RESULT_S2_DEBUG
		cout<<"mean_unnorm:"<<mean_unnorm<<endl;
		cout<<"var_unnorm:"<<var_unnorm<<endl;
	#endif


	
	ap_uint<ILN_N_VAR_WIDTH+4> var_shift=(ap_uint<ILN_N_VAR_WIDTH+4>(var_unnorm))<<4;

	#ifdef RESULT_S2_DEBUG
		cout<<"var_shift:"<<var_shift<<endl;
	#endif


	#ifdef RESULT_S2_DEBUG
		cout<<"ln_rcd_factor:"<<ln_rcd_factor<<endl;
	#endif

	mean_norm=(ap_ufixed<ILN_N_MEAN_WIDTH+8, ILN_N_MEAN_WIDTH>(mean_unnorm))/ln_rcd_factor;

	#ifdef RESULT_S2_DEBUG
		cout<<"mean_norm:"<<mean_norm<<endl;
	#endif

	var_norm= (ap_ufixed<ILN_N_VAR_WIDTH+4+8, ILN_N_VAR_WIDTH+4>(var_shift))/ln_rcd_factor;

	#ifdef RESULT_S2_DEBUG
		cout<<"var_norm:"<<var_norm<<endl;
	#endif

	ap_ufixed<ILN_VAR_WIDTH+8, ILN_VAR_WIDTH> std_inv_temp=var_norm - (mean_norm * mean_norm);


	#ifdef RESULT_S2_DEBUG
		cout<<"std_inv_temp:"<<std_inv_temp<<endl;
		cout<<"std_mean_temp:"<<mean_norm * mean_norm<<endl;
	#endif

	std_inv_temp=sqrt<ILN_VAR_WIDTH+8,ILN_VAR_WIDTH>(std_inv_temp);

	#ifdef RESULT_S2_DEBUG
		cout<<"std_inv_temp:"<<std_inv_temp<<endl;
	#endif

	 std_inv=ap_ufixed<18, 2>(1)/std_inv_temp;

	#ifdef RESULT_S2_DEBUG
		// cout<<"std_inv:"<<(ap_ufixed<10, 2>)(1)<<endl;
		cout<<"std_inv:"<<std_inv<<endl;
	#endif


}


void LNParam_Stream(
	ap_uint<ILN_WIDTH*2*2> ln_buf[MAX_OUP][MAX_NORM_INBUF_LENGTH],
	ap_uint<LN_PWF_FACTOR_BIT> buf_ln_ptf_factor[MAX_OUP][MAX_NORM_PWF_FACTOR_LENGTH],
	ap_uint<ILN_N_MEAN_WIDTH> mean_unnorm[MAX_INP][2],
	ap_uint<ILN_N_VAR_WIDTH> var_unnorm[MAX_INP][2],
	stream<ap_uint<ILN_OUT_WIDTH*2> > out[MAX_NORM_PE], 
	const unsigned MM_M,
	const unsigned PENUM,
	const unsigned OUT_W,
	const unsigned OUT_H,
	const unsigned group_num,
	bool tran_en,
	const unsigned Mode){
#pragma HLS INLINE OFF
//#pragma HLS BIND_STORAGE variable=ln_ptf_factor_buffer type=ram_2p impl=bram
	if (!tran_en) return;


#ifdef NROM_IN_DEBUG
    FILE* fp_in= fopen("norm_stage2_in.txt", "wb");
    FILE* fp_out= fopen("norm_stage2_out.txt", "wb");
#endif



	unsigned int numlines;
    // unsigned int outOupIdx=0;
	unsigned int outdIdx=0;
    unsigned int w=0;
    unsigned int h=0;
	unsigned int index;
	unsigned int index_ini_gb,index_gb;
	unsigned int index_ini_factor,index_factor;
	ap_uint<17> ln_rcd_factor;
//	cout<< index_factor_ini<<endl;

	unsigned int loop0,loop1,loop2,loop3;

	loop0=MAX_OUP/MAX_NORM_PE;

	if(Mode==0){  // MM
		ln_rcd_factor=MM_M;
		loop1=2;
		loop2=MAX_INP;
		loop3=PENUM; // M/(MAX_OUP*2)
		index_ini_gb=0;
		index_ini_factor=0;
		numlines= PENUM*MAX_INP*2;  // 这个有问题吗？
	}
	else if(Mode==1){
		ln_rcd_factor=(OUT_H*OUT_W*PENUM*MAX_OUP);
		loop1=OUT_W/2;
		loop2=PENUM;  // D/PENUM
		loop3=OUT_H;
		index_ini_gb=group_num*PENUM*(MAX_OUP/MAX_NORM_PE);
		index_ini_factor=(group_num/2)*PENUM;
		numlines= OUT_H*(PENUM)*(OUT_W/2);
	}


	ap_ufixed<ILN_MEAN_WIDTH+8, ILN_MEAN_WIDTH> mean_norm[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=mean_norm complete dim=2
	ap_ufixed<18, 2> std_inv[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=std_inv complete dim=2

	ap_ufixed<ILN_MEAN_WIDTH+8, ILN_MEAN_WIDTH> mean_norm_conv;
	ap_ufixed<18, 2> std_inv_conv;


	if(Mode==0){

		for(unsigned i=0; i<loop2;i++){
#pragma HLS UNROLL factor=2
			compute_mean_var(mean_unnorm[i][0],var_unnorm[i][0],ln_rcd_factor,mean_norm[i][0],std_inv[i][0]);
			compute_mean_var(mean_unnorm[i][1],var_unnorm[i][1],ln_rcd_factor,mean_norm[i][1],std_inv[i][1]);
		}

	}
	else if(Mode==1){

		ap_uint<ILN_N_MEAN_WIDTH> mean_unnorm_conv;
		ap_uint<ILN_N_VAR_WIDTH> var_unnorm_conv;
		mean_unnorm_conv=0;
		var_unnorm_conv=0;

		for(unsigned i=0; i<loop2;i++){
			mean_unnorm_conv=mean_unnorm_conv+mean_unnorm[i][0]+mean_unnorm[i][1];
			var_unnorm_conv=var_unnorm_conv+var_unnorm[i][0]+var_unnorm[i][0];
			
		}

		compute_mean_var(mean_unnorm_conv,var_unnorm_conv,ln_rcd_factor,mean_norm_conv,std_inv_conv);


	}



	LN_BIAS_DB gamma_temp, beta_temp,a_temp0,a_temp1;
	ap_uint<2> ptf_factor_temp;
	
	ap_uint<ILN_WIDTH*2> x0_16b,x1_16b; 
	ap_uint<ILN_WIDTH> x0,x1; 
	ap_uint<12> x0_shift,x1_shift; 
	LN_OUT_DB x0_mean,x1_mean; 

	ap_uint<ILN_OUT_WIDTH*2> res;

	ap_ufixed<ILN_MEAN_WIDTH+8, ILN_MEAN_WIDTH> mean_norm_temp0,mean_norm_temp1;

	ap_ufixed<ILN_VAR_WIDTH+8, ILN_VAR_WIDTH> var_norm_temp0,var_norm_temp1;

	ap_ufixed<18, 2> std_inv_temp0,std_inv_temp1;

	if(Mode==1){
		mean_norm_temp0=mean_norm_conv;
		mean_norm_temp1=mean_norm_conv;
		std_inv_temp0=std_inv_conv;
		std_inv_temp1=std_inv_conv;
	}


	#ifdef NROM_IN_DEBUG
		cout<<"mean_norm_temp0"<<mean_norm_temp0<<endl;
		cout<<"std_inv_temp0"<<std_inv_temp0<<endl;
	#endif


	for(int i=0;i<numlines;i++){
		for(unsigned g=0; g< MAX_OUP/MAX_NORM_PE;g++){
#pragma HLS PIPELINE II=1
			if(Mode==0){
				index_factor=loop1*h+w;
				index_gb=index_ini_gb+h*loop1*loop0+w*loop0+g;
				mean_norm_temp0=mean_norm[outdIdx][0];
				mean_norm_temp1=mean_norm[outdIdx][1];
				std_inv_temp0=std_inv[outdIdx][0];
				std_inv_temp1=std_inv[outdIdx][1];
			}
			else if(Mode==1){
				index_factor=index_ini_factor+outdIdx;
				index_gb=index_ini_gb+outdIdx*loop0+g;
			}
		
			for(int j=0;j<MAX_NORM_PE;j++){
				// obtain x
				(x1_16b,x0_16b)=ln_buf[g*MAX_NORM_PE+j][i];
				x0=x0_16b;
				x1=x1_16b;

				#ifdef PRINT_DEBUG
					cout<<"x0:"<<x0<<endl;
					cout<<"x1:"<<x1<<endl;
				#endif

				// obtain gamma, beta, ptf_factor

				gamma_temp=ln_gamma_buffer[j][index_gb];
				beta_temp=ln_beta_buffer[j][index_gb];
				ptf_factor_temp=buf_ln_ptf_factor[g*MAX_NORM_PE+j][index_factor];

				// cout<<"lnpeColIndex: "<<j<<"  index_input: "<<index<<endl;
				#ifdef PRINT_DEBUG
					cout<<"gamma_temp:"<<gamma_temp<<endl;
					cout<<"beta_temp:"<<beta_temp<<endl;
					cout<<"ptf_factor_temp:"<<ptf_factor_temp<<endl;
				#endif


				#ifdef NROM_IN_DEBUG
					// FILE* fp_parameter= fopen("norm_stage2.txt", "wb");
					fprintf(fp_in, "%lf\n", double(x0));
					fprintf(fp_in, "%lf\n", double(x1));
				#endif

				a_temp0=gamma_temp*std_inv_temp0;
				a_temp1=gamma_temp*std_inv_temp1;

				#ifdef RESULT_S2_DEBUG
					cout<<"a_temp:"<<a_temp<<endl;
				#endif

				x0_shift=(ap_uint<12>(x0)<<ptf_factor_temp);

				#ifdef RESULT_S2_DEBUG
					cout<<"x0_shift:"<<x0_shift<<endl;
					cout<<"mean_norm:"<<mean_norm<<endl;
				#endif

				x0_mean= x0_shift-mean_norm_temp0;

				#ifdef RESULT_S2_DEBUG
					cout<<"x0_mean:"<<x0_mean<<endl;
				#endif

				x0_mean=x0_mean*a_temp0+beta_temp;

				#ifdef RESULT_S2_DEBUG
					cout<<"x0_mean:"<<x0_mean<<endl;
				#endif

				x1_shift=(ap_uint<12>(x1)<<ptf_factor_temp);
				x1_mean= x1_shift-mean_norm_temp1;
				x1_mean=x1_mean*a_temp1+beta_temp;

				res=(x1_mean(ILN_OUT_WIDTH-1,0),x0_mean(ILN_OUT_WIDTH-1,0));
				out[j].write(res);

				#ifdef NROM_IN_DEBUG
					// FILE* fp_parameter= fopen("norm_stage2.txt", "wb");
					fprintf(fp_out, "%lf\n", double(res));
//					fprintf(fp_out, "%lf\n", double(x1_mean));
				#endif

			}
		}




		if(w==loop1-1){
			w=0;
			if(outdIdx==loop2-1){
				outdIdx=0;
				if(h==loop3-1){
					h=0;
				}
				else{
					h++;
				}
			}
			else{
				outdIdx++;
			}
		}
		else{
			w++;
		}


	}

#ifdef NROM_IN_DEBUG
    // FILE* fp_parameter= fopen("norm_stage2.txt", "wb");
	fclose(fp_in);
	fclose(fp_out);
#endif


}




void DDRReadShorcut(ap_uint<128> *input_shortcut_in,ap_uint<MAX_OUP*ILN_WIDTH*2> SHORTCUTBUF[MAX_NORM_INBUF_LENGTH],
	const unsigned PENUM,
	const unsigned OUT_W,
	const unsigned OUT_H,
	const unsigned group_num,
	const unsigned TOTAL_group,
	const unsigned SA_MODE,
	const bool FCU_MODE,
	const bool SHORTCUT_MODE){
#pragma HLS INLINE OFF

	if(SHORTCUT_MODE==false||group_num==TOTAL_group){
		return;
	}

	unsigned int numlines;
	unsigned int shortcut_index_ini;
	if(SA_MODE==0){  // MM
		numlines= PENUM*MAX_INP*2;
		shortcut_index_ini=group_num*numlines;
	}
	else if(SA_MODE==1&&FCU_MODE==0){

		numlines= OUT_H*(PENUM)*(OUT_W/2);
		shortcut_index_ini=group_num*numlines;
	}
	else if(SA_MODE==1&&FCU_MODE==1){

		numlines= (PENUM);
		shortcut_index_ini=group_num*numlines;
	}
	numlines=numlines*2;
	shortcut_index_ini=shortcut_index_ini*2;

	ap_uint<128> temp_128b[2];
	ap_uint<80> temp_80b[2];
	ap_uint<MAX_OUP*ILN_WIDTH*2> temp;

	 unsigned int bitIdx=0;
	 unsigned int depthIdx=0;

	for(int i=0;i<numlines;i++){
#pragma HLS PIPELINE II=1
		temp_128b[bitIdx]=input_shortcut_in[shortcut_index_ini+i];
		// cout<<temp<<endl;
		

        if(bitIdx==2-1){
			temp_80b[0]=temp_128b[0];
			temp_80b[1]=temp_128b[1];
            SHORTCUTBUF[depthIdx]=(temp_80b[1],temp_80b[0]);

			// temp=(temp_80b[1],temp_80b[0]);
			// cout <<"The Value of Var_p: \t" <<temp<< " \t Binary format: \t" <<temp.to_string(16).c_str()<< '\n';
        }

        if(bitIdx==2-1){
            bitIdx=0;
			if(depthIdx==MAX_NORM_INBUF_LENGTH-1){
				depthIdx=0;
			}
			else{
				depthIdx++;
			}
        }
        else{
            bitIdx++;
        }
	}
	
}





void LayerNorm (ap_uint<128> *input_shortcut_in,
	stream<ap_uint<DEQUAN_BIT*2> > LN_IN[MAX_OUP],
	stream<ap_uint<DEQUAN_BIT*2> > NO_LN_OUT[MAX_OUP],
    stream<ap_uint<ILN_OUT_WIDTH*2> > LN_OUT[MAX_NORM_PE],
	const unsigned MM_M,
	const unsigned PENUM,
	const unsigned OUT_W,
	const unsigned OUT_H,
	const unsigned GROUPS,
	const bool SA_MODE,
	const bool NORM_MODE,
	const bool SHORTCUT_MODE,
	const bool FCU_MODE){
#pragma HLS ALLOCATION instances=WriteLNParam limit=1 function
#pragma HLS ALLOCATION instances=LNParam_Stream limit=1 function
//#pragma HLS BIND_STORAGE variable=ln_ptf_factor_buffer type=ram_2p impl=bram
#pragma HLS INLINE OFF
	if(NORM_MODE==false){
		return;
	}


	ap_uint<ILN_N_MEAN_WIDTH> mean_unorm0[MAX_INP][2], mean_unorm1[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=mean_unorm0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=mean_unorm1 complete dim=2
	ap_uint<ILN_N_VAR_WIDTH> var_unnorm0[MAX_INP][2],var_unnorm1[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=var_unnorm0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=var_unnorm1 complete dim=2

	bool arb = 0;
	bool trans_en = 0;

	De_Quan_DB temp0,temp1;
	// scale_factor_buffer[4]
	temp0(DEQUAN_BIT-1,0)=scale_factor_buffer(63,48);



	// MUX_Shorcut_FCU(input_shortcut_in, SHORTCUT_IN_buffer0, PENUM, OUT_W, OUT_H,0,GROUPS,SA_MODE,SHORTCUT_MODE,FCU_MODE);

	DDRReadShorcut(input_shortcut_in, SHORTCUT_IN_buffer0, PENUM, OUT_W, OUT_H,0,GROUPS,SA_MODE,FCU_MODE, SHORTCUT_MODE);

	for(unsigned int i=0; i<GROUPS; i++){
		if(arb==0){
			WriteLNParam(LN_IN, SHORTCUT_IN_buffer0, LN_IN_buffer0, NO_LN_OUT, ln_ptf_factor_buffer0, temp0, mean_unorm0,var_unnorm0, PENUM, OUT_W, OUT_H,i,SA_MODE,SHORTCUT_MODE, FCU_MODE);
			// #ifdef RESULT_DEBUG
			// 	cout<<"mean_unnorm:"<<mean_unorm0<<endl;
			// 	cout<<"var_unnorm:"<<var_unnorm0<<endl;
			// #endif			
			LNParam_Stream(LN_IN_buffer1, ln_ptf_factor_buffer1, mean_unorm1,var_unnorm1,LN_OUT, MM_M, PENUM, OUT_W, OUT_H, i-1,trans_en,SA_MODE);
			// MUX_Shorcut_FCU(input_shortcut_in, SHORTCUT_IN_buffer1, PENUM, OUT_W, OUT_H,i+1,GROUPS,SA_MODE,SHORTCUT_MODE,FCU_MODE);
			DDRReadShorcut(input_shortcut_in, SHORTCUT_IN_buffer1, PENUM, OUT_W, OUT_H,i+1,GROUPS,SA_MODE,FCU_MODE, SHORTCUT_MODE);
		}
		else{

			WriteLNParam(LN_IN, SHORTCUT_IN_buffer1, LN_IN_buffer1, NO_LN_OUT, ln_ptf_factor_buffer1,temp0, mean_unorm1,var_unnorm1, PENUM, OUT_W, OUT_H,i,SA_MODE,SHORTCUT_MODE, FCU_MODE);
			LNParam_Stream(LN_IN_buffer0, ln_ptf_factor_buffer0, mean_unorm0,var_unnorm0,LN_OUT, MM_M, PENUM, OUT_W, OUT_H, i-1,trans_en,SA_MODE);
			// MUX_Shorcut_FCU(input_shortcut_in, SHORTCUT_IN_buffer0, PENUM, OUT_W, OUT_H,i+1,GROUPS,SA_MODE,SHORTCUT_MODE,FCU_MODE);
			DDRReadShorcut(input_shortcut_in, SHORTCUT_IN_buffer0, PENUM, OUT_W, OUT_H,i+1,GROUPS,SA_MODE,FCU_MODE, SHORTCUT_MODE);
		}
		trans_en = 1;
		arb = !arb;
	}

	if(arb==0){
		LNParam_Stream(LN_IN_buffer1,ln_ptf_factor_buffer1, mean_unorm1,var_unnorm1,LN_OUT, MM_M, PENUM, OUT_W, OUT_H, GROUPS-1,trans_en,SA_MODE);
	}
	else{
		LNParam_Stream(LN_IN_buffer0, ln_ptf_factor_buffer0, mean_unorm0,var_unnorm0,LN_OUT, MM_M, PENUM, OUT_W, OUT_H, GROUPS-1,trans_en,SA_MODE);
	}


}



void SiLU_Quan_Unit(stream<ap_uint<ILN_OUT_WIDTH * 2> > in[MAX_NORM_PE],
	stream<ap_uint<IN_BIT * 2> > out[MAX_NORM_PE],
	const unsigned NumLines,
	const bool NORM_MODE,
	const bool SA_MODE){

	if((NORM_MODE)==false){
		return;
	}

	Quan_Factor_DB quan_factor_temp;
	//scale_factor_buffer[1]=temp_scale_factor(31,16);
	quan_factor_temp(QUAN_FACTOR_BIT-1,0)=scale_factor_buffer(31,16);



	ap_uint<ILN_OUT_WIDTH> x0,x1;
	ap_fixed<ILN_OUT_WIDTH,ILN_OUT_INTEGER_WIDTH> fixp_x0,fixp_x1;

	ap_fixed<SILU_BIT,SILU_INTEGER_BIT> out0,out1;
	ap_int<IN_BIT> res0,res1;

	const ap_fixed<16, 2> onedivsixth = 0.16666666; 


	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_NORM_PE; c++){
		ap_uint<ILN_OUT_WIDTH * 2> temp = in[c].read();
		(x1,x0)=temp;
		fixp_x0(ILN_OUT_WIDTH-1,0)=x0(ILN_OUT_WIDTH-1,0);
		fixp_x1(ILN_OUT_WIDTH-1,0)=x1(ILN_OUT_WIDTH-1,0);

		if(SA_MODE==true){
		out0=compute_silu<ILN_OUT_WIDTH,ILN_OUT_INTEGER_WIDTH,SILU_BIT,SILU_INTEGER_BIT>(fixp_x0);
		out1=compute_silu<ILN_OUT_WIDTH,ILN_OUT_INTEGER_WIDTH,SILU_BIT,SILU_INTEGER_BIT>(fixp_x1);
		}
		else{
		out0=fixp_x0;
		out1=fixp_x1;
		}
		res0=(ap_fixed<OUT_BIT,OUT_BIT>)(out0*quan_factor_temp);
		res1=(ap_fixed<OUT_BIT,OUT_BIT>)(out1*quan_factor_temp);

		out[c].write((res1,res0));
	  }
	}

}


void Softmax_Quan_Unit(stream<ap_uint<SOFTMAX_OUT_WIDTH * 2> > in[MAX_SOFTMAX_STAGE2_PE],
	stream<ap_uint<OUT_BIT * 2> > out[MAX_SOFTMAX_STAGE2_PE],
	const unsigned NumLines,
	const bool SOFTMAX_MODE){

	if(SOFTMAX_MODE==false){
		return;
	}

	ap_uint<SOFTMAX_OUT_WIDTH> x0,x1;
	ap_fixed<SOFTMAX_OUT_WIDTH,SOFTMAX_OUT_INTEGER_WIDTH> fixp_x0,fixp_x1;

	ap_int<OUT_BIT> res0,res1;
	Quan_Factor_DB quan_factor_temp;
	// scale_factor_buffer[5]=temp_scale_factor(95,80);
	quan_factor_temp(QUAN_FACTOR_BIT-1,0)=scale_factor_buffer(31,16);


	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_SOFTMAX_STAGE2_PE; c++){
		ap_uint<SOFTMAX_OUT_WIDTH * 2> temp = in[c].read();
		(x1,x0)=temp;
		fixp_x0(SOFTMAX_OUT_WIDTH-1,0)=x0(SOFTMAX_OUT_WIDTH-1,0);
		fixp_x1(SOFTMAX_OUT_WIDTH-1,0)=x1(SOFTMAX_OUT_WIDTH-1,0);
		res0=(ap_fixed<OUT_BIT,OUT_BIT>)(fixp_x0*quan_factor_temp);
		res1=(ap_fixed<OUT_BIT,OUT_BIT>)(fixp_x1*quan_factor_temp);
		out[c].write((res1,res0));
	  }
	}

}



void Gelu_Quan_Unit(stream<ap_uint<GELU_OUT_WIDTH * 2> > in[MAX_GELU_PE],
	stream<ap_uint<OUT_BIT * 2> > out[MAX_GELU_PE],
	const unsigned NumLines,
	const bool GELU_MODE){

	if(GELU_MODE==false){
		return;
	}

	ap_uint<GELU_OUT_WIDTH> x0,x1;
	ap_fixed<GELU_OUT_WIDTH,GELU_OUT_INTEGER_WIDTH> fixp_x0,fixp_x1;

	ap_int<OUT_BIT> res0,res1;
	Quan_Factor_DB quan_factor_temp;
	// scale_factor_buffer[5]=temp_scale_factor(95,80);
	quan_factor_temp(QUAN_FACTOR_BIT-1,0)=scale_factor_buffer(31,16);


	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_SOFTMAX_STAGE2_PE; c++){
		ap_uint<GELU_OUT_WIDTH * 2> temp = in[c].read();
		(x1,x0)=temp;
		fixp_x0(GELU_OUT_WIDTH-1,0)=x0(GELU_OUT_WIDTH-1,0);
		fixp_x1(GELU_OUT_WIDTH-1,0)=x1(GELU_OUT_WIDTH-1,0);
		res0=(ap_fixed<OUT_BIT,OUT_BIT>)(fixp_x0*quan_factor_temp);
		res1=(ap_fixed<OUT_BIT,OUT_BIT>)(fixp_x1*quan_factor_temp);
		out[c].write((res1,res0));
	  }
	}

}

void No_Emulti_Quan_Unit(stream<ap_uint<DEQUAN_BIT * 2> > in[MAX_OUP],
	stream<ap_uint<OUT_BIT * 2> > out[MAX_OUP],
	const unsigned NumLines,
	const bool EBMULT_MODE){

	if(EBMULT_MODE==false){
		return;
	}

	ap_uint<DEQUAN_BIT> x0,x1;
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> fixp_x0,fixp_x1;

	ap_int<OUT_BIT> res0,res1;
	Quan_Factor_DB quan_factor_temp;
	// scale_factor_buffer[5]=temp_scale_factor(95,80);
	quan_factor_temp(QUAN_FACTOR_BIT-1,0)=scale_factor_buffer(31,16);


	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_OUP; c++){
		ap_uint<DEQUAN_BIT * 2> temp = in[c].read();
		(x1,x0)=temp;
		fixp_x0(DEQUAN_BIT-1,0)=x0(DEQUAN_BIT-1,0);
		fixp_x1(DEQUAN_BIT-1,0)=x1(DEQUAN_BIT-1,0);
		res0=(ap_fixed<OUT_BIT,OUT_BIT>)(fixp_x0*quan_factor_temp);
		res1=(ap_fixed<OUT_BIT,OUT_BIT>)(fixp_x1*quan_factor_temp);
		out[c].write((res1,res0));
	  }
	}

}

void ShortcutQuanUnit(stream<ap_uint<DEQUAN_BIT * 2> > in[MAX_OUP],
	stream<ap_uint<IN_BIT * 2 *MAX_OUP> > &out,
	const unsigned NumLines,
	const bool SHORCUT_QUAN_MODE,
	const bool SHORCUT_ADD_MODE,
	const bool FCU_MODE
	){

	if(SHORCUT_QUAN_MODE==false){
		return;
	}

	ap_uint<DEQUAN_BIT> x0,x1;
	De_Quan_DB fixp_x0,fixp_x1;
	Quan_Factor_DB quan_factor_temp;
	ap_fixed<IN_BIT,IN_BIT> out0,out1;
	ap_uint<IN_BIT> res0,res1;
	ap_uint<IN_BIT * 2 *MAX_OUP> out_temp;

	//scale_factor_buffer[2]=temp_scale_factor(47,32);
	quan_factor_temp(QUAN_FACTOR_BIT-1,0)=scale_factor_buffer(47,32);

//	cout<<"Layer_Scale: "<<quan_factor_temp<<endl;

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
	
      for(unsigned int c = 0; c < MAX_OUP; c++){
		ap_uint<DEQUAN_BIT * 2> temp = in[c].read();
		(x1,x0)=temp;
		fixp_x0(DEQUAN_BIT-1,0)=x0(DEQUAN_BIT-1,0);
		fixp_x1(DEQUAN_BIT-1,0)=x1(DEQUAN_BIT-1,0);
		out0=(ap_fixed<IN_BIT,IN_BIT>)(fixp_x0*quan_factor_temp);
		out1=(ap_fixed<IN_BIT,IN_BIT>)(fixp_x1*quan_factor_temp);
		res0=out0;
		res1=out1;
		out_temp=out_temp>>IN_BIT * 2;
		out_temp(IN_BIT * 2 *MAX_OUP-1,IN_BIT * 2 *MAX_OUP-IN_BIT * 2)=(res1,res0);

	  }

	  out.write(out_temp);
	}

}



void Nonlinear_QuanUnit(stream<ap_uint<IN_BIT * 2> > in[MAX_OUP],
	stream<ap_uint<IN_BIT * 2 *MAX_OUP> > &out,
	const unsigned NumLines,
	const bool NON_LINEAR_QUAN_MODE
	){

	if(NON_LINEAR_QUAN_MODE==false){
		return;
	}

	ap_uint<IN_BIT * 2 *MAX_OUP> out_temp;

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
	
      for(unsigned int c = 0; c < MAX_OUP; c++){
		ap_uint<IN_BIT * 2> temp = in[c].read();
		out_temp(IN_BIT * 2 *(c+1)-1,IN_BIT * 2 *c)=temp;
	  }


	  out.write(out_temp);
	}

}



void SOFTMAX_WriteTwoTimes(stream<ap_uint<DEQUAN_BIT*2> > in[MAX_OUP], 
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_SOFTMAX_INBUF_LENGTH],
	stream<ap_uint<DEQUAN_BIT*2> > NO_SOFTMAX_OUT[MAX_OUP],
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> tmax_M[MAX_INP][2],
	const unsigned PENUM,
	const bool EBMULT_MODE){
#pragma HLS INLINE OFF

	SOFTMAX_WriteBUF<DEQUAN_BIT,MAX_INP,MAX_OUP,MAX_NORM_INBUF_LENGTH>(in,ROW_T_buf,PENUM);
	SOFTMAX_WriteBUF_ADDBUF<DEQUAN_BIT,DEQUAN_INTEGER_BIT,MAX_INP,MAX_OUP,MAX_NORM_INBUF_LENGTH>(in,ROW_T_buf,NO_SOFTMAX_OUT, tmax_M, PENUM,EBMULT_MODE);
	

	#ifdef DEBUG
		(fp_temp1,fp_temp0)=ROW_T_buf[0];
		temp_x0_tran(DEQUAN_BIT-1,0)=fp_temp0(DEQUAN_BIT-1,0); 
		temp_x1_tran(DEQUAN_BIT-1,0)=fp_temp1(DEQUAN_BIT-1,0);

		cout<<"temp_x0_tran:"<<temp_x0_tran<<endl;
		cout<<"temp_x1_tran:"<<temp_x1_tran<<endl;
	#endif
}



void SOFTMAX_STAGE1_FUN(
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_SOFTMAX_INBUF_LENGTH],
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> tmax_L[2],
	ap_uint<DEQUAN_BIT*2> ONE_ROW_buf[MAX_SOFTMAX_STAGE2_PE][MAX_SOFTMAX_M_LENGTH],
	
	SOFTMAX_SUM_DB Sum_buf[2],
	const unsigned INP_NUM,
	const unsigned PENUM
	){
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=ROW_T_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=ONE_ROW_buf dim=1 complete
#pragma HLS ARRAY_PARTITION variable=tmax_L complete dim=0
#pragma HLS ARRAY_PARTITION variable=Sum_buf complete dim=0


	ap_uint< DEQUAN_BIT*2> temp;

	ap_uint<DEQUAN_BIT> temp_x0, temp_x1;

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp_x0_fixp, temp_x1_fixp;



	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> max_temp0;
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> max_temp1;

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> y_input0,y_input1;

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT>  y_out0,y_out1;

	SOFTMAX_SUM_DB  Sum_out_TempBuf[2];
#pragma HLS ARRAY_PARTITION variable=Sum_out_TempBuf complete dim=0



	SOFTMAX_SUM_DB sum_temp0,sum_temp1;





	unsigned int numlines;
	unsigned int ini_addr;




	numlines= PENUM*2;

	ini_addr= INP_NUM*numlines;



	Sum_out_TempBuf[0]=0;
	Sum_out_TempBuf[1]=0;



	for(unsigned m=0; m<numlines;m++){
		for(unsigned g0=0; g0< MAX_OUP/MAX_SOFTMAX_STAGE2_PE;g0++){
			for(int g1=0;g1<MAX_SOFTMAX_STAGE2_PE/MAX_SOFTMAX_STAGE1_PE;g1++){
#pragma HLS PIPELINE II=1
				for(int j=0;j<MAX_SOFTMAX_STAGE1_PE;j++){
					temp = ROW_T_buf[g0*MAX_SOFTMAX_STAGE2_PE+g1*MAX_SOFTMAX_STAGE1_PE+j][ini_addr+m];
					// cout<<temp<<endl;
					// 存入BRAM

					(temp_x1,temp_x0)=temp;

					temp_x0_fixp(DEQUAN_BIT-1,0)=temp_x0(DEQUAN_BIT-1,0);  // X_i
					temp_x1_fixp(DEQUAN_BIT-1,0)=temp_x1(DEQUAN_BIT-1,0);

					#ifdef STAGE1_DEBUG
						cout<<"temp_x0_fixp:"<<temp_x0_fixp<<endl;
						cout<<"temp_x1_fixp:"<<temp_x1_fixp<<endl;
					#endif

			
					max_temp0=tmax_L[0];
					y_input0=temp_x0_fixp-max_temp0;
					y_out0=exp<DEQUAN_BIT,DEQUAN_INTEGER_BIT>(y_input0);



					#ifdef STAGE1_DEBUG
						cout<<"max_temp0: "<<max_temp0<<endl; 
						cout<<"y_input0: "<<y_input0<<endl; 
						cout<<"y_out0: "<<y_out0<<endl; 
						cout<<"Sum_out_TempBuf[0]:"<<Sum_out_TempBuf[0]<<endl; 
					
					#endif	

					Sum_out_TempBuf[0]=Sum_out_TempBuf[0]+y_out0;
					
					#ifdef STAGE1_DEBUG

						cout<<"Sum_out_TempBuf[0]:"<<Sum_out_TempBuf[0]<<endl; 
					
					#endif	


					max_temp1=tmax_L[1];
					y_input1=temp_x1_fixp-max_temp1;
					y_out1=exp<DEQUAN_BIT,DEQUAN_INTEGER_BIT>(y_input1);



					#ifdef STAGE1_DEBUG
						cout<<"max_temp1:"<<max_temp1<<endl; 
						cout<<"y_input1:"<<y_input1<<endl; 
						cout<<"y_out1:"<<y_out1<<endl; 
						cout<<"Sum_out_TempBuf[1]:"<<Sum_out_TempBuf[1]<<endl; 
					#endif	

					Sum_out_TempBuf[1]=Sum_out_TempBuf[1]+y_out1;

					#ifdef STAGE1_DEBUG
						cout<<"Sum_out_TempBuf[1]:"<<Sum_out_TempBuf[1]<<endl; 
					#endif	



					ONE_ROW_buf[g1*MAX_SOFTMAX_STAGE1_PE+j][m*(MAX_OUP/MAX_SOFTMAX_STAGE2_PE)+g0]=((y_out1(DEQUAN_BIT-1,0),y_out0(DEQUAN_BIT-1,0)));
					// Sum_out_OUPBuf[0][j]=y_out0;
					// Sum_out_OUPBuf[1][j]=y_out1;

				}	
			}
		}

		// for(int j=0;j<MAX_SOFTMAX_STAGE1_PE;j++){
		// 	ROW_T_buf[m*MAX_SOFTMAX_STAGE1_PE+j]=((Sum_out_OUPBuf[1][j](DEQUAN_BIT-1,0),Sum_out_OUPBuf[0][j](DEQUAN_BIT-1,0)));
		// }




	}


	Sum_buf[0]=Sum_out_TempBuf[0];
	Sum_buf[1]=Sum_out_TempBuf[1];


}



void SOFTMAX_STAGE2_FUN(
	ap_uint<DEQUAN_BIT*2> ONE_ROW_buf[MAX_SOFTMAX_STAGE2_PE][MAX_SOFTMAX_M_LENGTH],
	SOFTMAX_SUM_DB Sum_buf[2],
	stream<ap_uint<SOFTMAX_OUT_WIDTH*2> > res_out[MAX_SOFTMAX_STAGE2_PE], 
	const unsigned PENUM,
	bool tran_en
	){
#pragma HLS INLINE OFF
	if (!tran_en) return;

#pragma HLS ARRAY_PARTITION variable=Sum_buf complete dim=0

	ap_uint< DEQUAN_BIT*2> temp;

	ap_uint<DEQUAN_BIT> temp_x0, temp_x1;

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp_x0_y, temp_x1_y;



	unsigned int outdIdx=0;
    unsigned int w=0;
    unsigned int h=0;
	unsigned int g=0;
	unsigned int outOupIdx=0;

	unsigned int numlines;

	int cnt=0;
	SOFTMAX_SUM_DB sum_temp0,sum_temp1;


	// MAX_OUP/MAX_SOFTMAX_STAGE2_PE
	numlines= PENUM*2;


	ap_fixed<SOFTMAX_OUT_WIDTH, SOFTMAX_OUT_INTEGER_WIDTH> out0,out1;


	// FILE* fp1_test = fopen("result_verify_M.txt", "wb");
	sum_temp0=Sum_buf[0];
	sum_temp1=Sum_buf[1];

	for(unsigned m=0; m<numlines;m++){
		for(unsigned g=0; g< MAX_OUP/MAX_SOFTMAX_STAGE2_PE;g++){
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE false inter variable=ONE_ROW_buf
			for(int j=0;j<MAX_SOFTMAX_STAGE2_PE;j++){


				temp = ONE_ROW_buf[j][m*(MAX_OUP/MAX_SOFTMAX_STAGE2_PE)+g];

				(temp_x1, temp_x0)=temp;
				temp_x0_y(DEQUAN_BIT-1,0)=temp_x0(DEQUAN_BIT-1,0);  // X_i
				temp_x1_y(DEQUAN_BIT-1,0)=temp_x1(DEQUAN_BIT-1,0);

				#ifdef STAGE2_DEBUG
					cout<<"temp_x0_y:"<<temp_x0_y<<endl;
					cout<<"temp_x1_y:"<<temp_x1_y<<endl;
				#endif

				#ifdef STAGE2_DEBUG
					cout<<"sum_temp0:"<<sum_temp0<<endl;
					cout<<"sum_temp1:"<<sum_temp1<<endl;
				#endif

				out0=(ap_fixed<DEQUAN_BIT+8, DEQUAN_INTEGER_BIT>(temp_x0_y))/sum_temp0;
				out1=(ap_fixed<DEQUAN_BIT+8, DEQUAN_INTEGER_BIT>(temp_x1_y))/sum_temp1;
				

				#ifdef STAGE2_DEBUG
					cout<<"out0:"<<out0<<endl;
					cout<<"out1:"<<out1<<endl;
				#endif

				res_out[j].write((out1(SOFTMAX_OUT_WIDTH-1,0),out0(SOFTMAX_OUT_WIDTH-1,0)));

			}
		}

	}

	// fclose(fp1_test);
}

void SOFTMAX_WriteStream(
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_SOFTMAX_INBUF_LENGTH],
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> tmax_L[MAX_INP][2],
	stream<ap_uint<SOFTMAX_OUT_WIDTH*2> > res_out[MAX_SOFTMAX_STAGE2_PE], 
	const unsigned PENUM,
	bool tran_en){
#pragma HLS INLINE OFF
	
	if (!tran_en) return;



	ap_uint<DEQUAN_BIT*2> ONE_ROW_buf_ping[MAX_SOFTMAX_STAGE2_PE][MAX_SOFTMAX_M_LENGTH];  // 值应该不等于MAX_SOFTMAX_INBUF_LENGTH
#pragma HLS ARRAY_PARTITION variable=ONE_ROW_buf_ping dim=1 complete
	ap_uint<DEQUAN_BIT*2> ONE_ROW_buf_pong[MAX_SOFTMAX_STAGE2_PE][MAX_SOFTMAX_M_LENGTH];
#pragma HLS ARRAY_PARTITION variable=ONE_ROW_buf_pong dim=1 complete

	SOFTMAX_SUM_DB Sum_buf_ping[2];
#pragma HLS ARRAY_PARTITION variable=Sum_buf_ping complete dim=0
	SOFTMAX_SUM_DB Sum_buf_pong[2];
#pragma HLS ARRAY_PARTITION variable=Sum_buf_pong complete dim=0



	bool arb = 0;
	bool intra_trans_en = 0;

	for(unsigned int i=0; i<MAX_INP; i++){
		if(arb==0){
			SOFTMAX_STAGE1_FUN(ROW_T_buf, tmax_L[i], ONE_ROW_buf_ping,Sum_buf_ping,  i, PENUM);

			SOFTMAX_STAGE2_FUN(ONE_ROW_buf_pong, Sum_buf_pong, res_out,PENUM, intra_trans_en);
		}
		else{

			SOFTMAX_STAGE1_FUN(ROW_T_buf, tmax_L[i], ONE_ROW_buf_pong,Sum_buf_pong,  i, PENUM);
			SOFTMAX_STAGE2_FUN(ONE_ROW_buf_ping, Sum_buf_ping, res_out,PENUM, intra_trans_en);

		}
		intra_trans_en = 1;
		arb = !arb;
	}

	if(arb==0){
		SOFTMAX_STAGE2_FUN(ONE_ROW_buf_pong, Sum_buf_pong, res_out,PENUM, intra_trans_en);
	}
	else{
		SOFTMAX_STAGE2_FUN(ONE_ROW_buf_ping, Sum_buf_ping, res_out,PENUM, intra_trans_en);
	}

}


void SOFTMAX_UNIT(stream<ap_uint<DEQUAN_BIT * 2> > SOFTMAX_IN[MAX_OUP],
	stream<ap_uint<DEQUAN_BIT*2> > NO_SOFTMAX_OUT[MAX_OUP], 
	stream<ap_uint<SOFTMAX_OUT_WIDTH*2> > SOFTMAX_OUT[MAX_SOFTMAX_STAGE2_PE],
	const unsigned PENUM,
	const unsigned GROUPS,
	const bool SOFTMAX_MODE,
	const bool EBMULT_MODE){
#pragma HLS INLINE OFF
	int softmax_PENUM;

	// if(SOFTMAX_MODE){
	// 	softmax_PENUM=PENUM>>1;
	// }
	// else{
		softmax_PENUM=PENUM>>1;		
	// }

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> tmax_M_ping[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=tmax_M_ping complete dim=2
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> tmax_M_pong[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=tmax_M_pong complete dim=2

	bool arb = 0;
	bool trans_en = 0;

	if(SOFTMAX_MODE==true){
		for(unsigned int i=0; i<GROUPS; i++){
			if(arb==0){
				SOFTMAX_WriteTwoTimes(SOFTMAX_IN, LN_IN_buffer0, NO_SOFTMAX_OUT, tmax_M_ping, softmax_PENUM,EBMULT_MODE);

				SOFTMAX_WriteStream(LN_IN_buffer1, tmax_M_pong, SOFTMAX_OUT,softmax_PENUM, trans_en);
			}
			else{
				SOFTMAX_WriteTwoTimes(SOFTMAX_IN, LN_IN_buffer1, NO_SOFTMAX_OUT, tmax_M_pong, softmax_PENUM,EBMULT_MODE);

				SOFTMAX_WriteStream(LN_IN_buffer0, tmax_M_ping, SOFTMAX_OUT,softmax_PENUM, trans_en);
			}
			trans_en = 1;
			arb = !arb;
		}

		if(arb==0){
			SOFTMAX_WriteStream(LN_IN_buffer1, tmax_M_pong, SOFTMAX_OUT, softmax_PENUM,trans_en);
		}
		else{
			SOFTMAX_WriteStream(LN_IN_buffer0, tmax_M_ping, SOFTMAX_OUT,softmax_PENUM, trans_en);
		}
	}
	else if(EBMULT_MODE==true){

		for(unsigned int i=0; i<GROUPS; i++){
			SOFTMAX_WriteTwoTimes(SOFTMAX_IN, LN_IN_buffer0, NO_SOFTMAX_OUT, tmax_M_ping, softmax_PENUM,EBMULT_MODE);
		}





	}


}



void GeLU_WriteBUF(stream<ap_uint<DEQUAN_BIT*2> > in[MAX_OUP], 
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_GELU_INBUF_LENGTH],
	const unsigned PENUM
	){
#pragma HLS INLINE OFF


	unsigned int numlines;

	unsigned int outdIdx=0;
    unsigned int w=0;
    unsigned int h=0;


	numlines= PENUM*MAX_INP*2;
	ap_uint< DEQUAN_BIT*2> temp;

	for(unsigned m=0; m<numlines;m++){
#pragma HLS PIPELINE II=1
		for(unsigned i=0; i<MAX_OUP;i++){
			temp = in[i].read();


			#ifdef RESULT_DEBUG
				ap_uint<DEQUAN_BIT> temp_x0, temp_x1;
				ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp_x0_fixp, temp_x1_fixp;
				(temp_x1,temp_x0)=temp;

				temp_x0_fixp(DEQUAN_BIT-1,0)=temp_x0(DEQUAN_BIT-1,0); 
				temp_x1_fixp(DEQUAN_BIT-1,0)=temp_x1(DEQUAN_BIT-1,0);

				
					cout<<"temp_x0_fixp:"<<temp_x0_fixp<<endl; 
					cout<<"temp_x1_fixp:"<<temp_x1_fixp<<endl; 
			#endif



			ROW_T_buf[i][outdIdx*PENUM*2+h*2+w]=temp;
		}

		if(w==2-1){
			w=0;
			if(outdIdx==MAX_INP-1){
				outdIdx=0;
				if(h==PENUM-1){
					h=0;
				}
				else{
					h++;
				}
			}
			else{
				outdIdx++;
			}
		}
		else{
			w++;
		}

	}

}



void GeLU_WriteStream(
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_GELU_INBUF_LENGTH],
	stream<ap_uint<GELU_OUT_WIDTH*2> > res_out[MAX_GELU_PE],
	const unsigned PENUM,
	bool tran_en
	){


	if (!tran_en) return;

	unsigned int numlines;

	// unsigned int outOupIdx=0;
	unsigned int outdIdx=0;
    unsigned int w=0;
    unsigned int h=0;
	unsigned int cnt_write=0,cnt_read=0;


	numlines= MAX_INP*PENUM*2;
	ap_uint< DEQUAN_BIT*2> temp;
	// ap_uint< DEQUAN_BIT*2> temp_buf;

	ap_uint<DEQUAN_BIT> x0,x1;
	ap_fixed<DEQUAN_BIT,DEQUAN_INTEGER_BIT> fixp_x0,fixp_x1;

	ap_uint<DEQUAN_BIT> x0_buf,x1_buf;
	ap_fixed<DEQUAN_BIT,DEQUAN_INTEGER_BIT> fixp_x0_buf,fixp_x1_buf;

	ap_fixed<GELU_OUT_WIDTH,GELU_OUT_INTEGER_WIDTH> out0,out1;

	ap_fixed<GELU_OUT_WIDTH,GELU_OUT_INTEGER_WIDTH> res0,res1;

	ap_uint<DEQUAN_BIT*2> HALF_ROW_buf[MAX_GELU_PE][MAX_GELU_ROW_INBUF_LENGTH/2];  // 值应该不等于MAX_SOFTMAX_INBUF_LENGTH
#pragma HLS ARRAY_PARTITION variable=HALF_ROW_buf dim=1 complete



	for(unsigned m=0; m<numlines;m++){
		for(unsigned g=0; g< MAX_OUP/MAX_GELU_PE;g++){
#pragma HLS PIPELINE II=1
			for(unsigned i=0; i<MAX_GELU_PE;i++){

				temp=ROW_T_buf[g*MAX_GELU_PE+i][m];



				if(h<(PENUM/2)){

					
					HALF_ROW_buf[i][h*2*(MAX_OUP/MAX_GELU_PE)+w*(MAX_OUP/MAX_GELU_PE)+g]=temp; 


					#ifdef GELU_DEBUG
						cout<<"cnt read set to 0: "<<cnt_read<<endl;
						cout<<"大buffer读数索引: "<<m*MAX_GELU_PE+i<<endl;
						cout<<"cnt write 索引: "<<cnt_write<<endl;

						// (x1,x0)=temp;
						// fixp_x0(DEQUAN_BIT-1,0)=x0(DEQUAN_BIT-1,0);
						// fixp_x1(DEQUAN_BIT-1,0)=x1(DEQUAN_BIT-1,0);		

						// cout<<"fixp_x0: "<<fixp_x0<<endl;
						// cout<<"fixp_x1: "<<fixp_x1<<endl;

						cout<<".........................."<<endl;
					#endif



				}
				else{
					(x1_buf,x0_buf)=HALF_ROW_buf[i][(h-PENUM/2)*2*(MAX_OUP/MAX_GELU_PE)+w*(MAX_OUP/MAX_GELU_PE)+g];

					fixp_x0_buf(DEQUAN_BIT-1,0)=x0_buf(DEQUAN_BIT-1,0);
					fixp_x1_buf(DEQUAN_BIT-1,0)=x1_buf(DEQUAN_BIT-1,0);				

					(x1,x0)=temp;
					fixp_x0(DEQUAN_BIT-1,0)=x0(DEQUAN_BIT-1,0);
					fixp_x1(DEQUAN_BIT-1,0)=x1(DEQUAN_BIT-1,0);			


					out0=compute_gelu<GELU_OUT_WIDTH,GELU_OUT_INTEGER_WIDTH,GELU_OUT_WIDTH,GELU_OUT_INTEGER_WIDTH>(fixp_x0);
					out1=compute_gelu<GELU_OUT_WIDTH,GELU_OUT_INTEGER_WIDTH,GELU_OUT_WIDTH,GELU_OUT_WIDTH>(fixp_x1);

					res0=out0+fixp_x0_buf;
					res1=out1+fixp_x1_buf;

					#ifdef GELU_DEBUG
						cout<<"cnt write set to 0: "<<cnt_write<<endl;
						cout<<"大buffer读数索引: "<<m*MAX_GELU_PE+i<<endl;
						cout<<"cnt read 索引: "<<cnt_read<<endl;

						// cout<<"fixp_x0_buf: "<<fixp_x0_buf<<endl;
						// cout<<"out0: "<<out0<<endl;
						// cout<<"res0: "<<res0<<endl;

						// cout<<"fixp_x1_buf: "<<fixp_x1_buf<<endl;
						// cout<<"out1: "<<out1<<endl;
						// cout<<"res1: "<<res1<<endl;

						cout<<".................................."<<endl;
					#endif
						res_out[i].write((res1(GELU_OUT_WIDTH-1,0),res0(GELU_OUT_WIDTH-1,0)));




				}

			}
		

		
		}

		if(w==2-1){
			w=0;
			if(h==PENUM-1){
				h=0;
				if(outdIdx==MAX_INP-1){
					outdIdx=0;
				}
				else{
					outdIdx++;
				}
			}
			else{
				h++;
			}
		}
		else{
			w++;
		}



	}

}



void Tranpose_WriteStream(
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_GELU_INBUF_LENGTH],
	stream<ap_uint<GELU_OUT_WIDTH*2> > res_out[MAX_GELU_PE],
	const unsigned PENUM,
	bool tran_en
	){
#pragma HLS INLINE OFF


	if (!tran_en) return;

	unsigned int numlines;

	unsigned int outOupIdx=0;
	unsigned int outdIdx=0;
    unsigned int w=0;
    unsigned int h=0;
	unsigned int cnt_write=0,cnt_read=0;


	numlines= MAX_INP*PENUM*2;
	ap_uint< DEQUAN_BIT*2> temp;

	for(unsigned m=0; m<numlines;m++){
		for(unsigned g=0; g< MAX_OUP/MAX_GELU_PE;g++){
#pragma HLS PIPELINE II=1
			for(unsigned i=0; i<MAX_GELU_PE;i++){

				temp=ROW_T_buf[g*MAX_GELU_PE+i][m];

				#ifdef GELU_DEBUG
					cout<<"cnt write set to 0: "<<cnt_write<<endl;
					cout<<"大buffer读数索引: "<<m*MAX_GELU_PE+i<<endl;
					cout<<"cnt read 索引: "<<cnt_read<<endl;

					// cout<<"fixp_x0_buf: "<<fixp_x0_buf<<endl;
					// cout<<"out0: "<<out0<<endl;
					// cout<<"res0: "<<res0<<endl;

					// cout<<"fixp_x1_buf: "<<fixp_x1_buf<<endl;
					// cout<<"out1: "<<out1<<endl;
					// cout<<"res1: "<<res1<<endl;

					cout<<".................................."<<endl;
				#endif
				res_out[i].write(temp);

			}
		}

	}

}


void MUX_GeLU_TRANPOSE_WriteStream(
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_GELU_INBUF_LENGTH],
	stream<ap_uint<GELU_OUT_WIDTH*2> > res_out[MAX_GELU_PE],
	const unsigned PENUM,
	bool tran_en,
	const bool GELU_MODE,
	const bool TRANPOSE_MODE){
#pragma HLS INLINE OFF
	if (!tran_en) return;
	
	if(GELU_MODE){
		GeLU_WriteStream(ROW_T_buf, res_out, PENUM, tran_en);
	}
	else if(TRANPOSE_MODE){
		Tranpose_WriteStream(ROW_T_buf, res_out, PENUM, tran_en);
	}

}

void GeLU_TranPose_UNIT (stream<ap_uint<DEQUAN_BIT*2> > GeLU_IN[MAX_OUP],
    stream<ap_uint<GELU_OUT_WIDTH*2> > GeLU_OUT[MAX_GELU_PE],
	const unsigned PENUM,
	const unsigned GROUPS,
	const bool GELU_MODE,
	const bool TRANPOSE_MODE){

	if(GELU_MODE==false&&TRANPOSE_MODE==false){
		return;
	}

	bool arb = 0;
	bool trans_en = 0;

	for(unsigned int i=0; i<GROUPS; i++){
		if(arb==0){
			GeLU_WriteBUF(GeLU_IN, LN_IN_buffer0, PENUM);
			MUX_GeLU_TRANPOSE_WriteStream(LN_IN_buffer1, GeLU_OUT, PENUM, trans_en,GELU_MODE,TRANPOSE_MODE);
			
		}
		else{
			GeLU_WriteBUF(GeLU_IN, LN_IN_buffer1, PENUM);

			MUX_GeLU_TRANPOSE_WriteStream(LN_IN_buffer0, GeLU_OUT, PENUM, trans_en,GELU_MODE,TRANPOSE_MODE);
		}
		trans_en = 1;
		arb = !arb;
	}

	if(arb==0){
		MUX_GeLU_TRANPOSE_WriteStream(LN_IN_buffer1, GeLU_OUT, PENUM, trans_en,GELU_MODE,TRANPOSE_MODE);
	}
	else{
		MUX_GeLU_TRANPOSE_WriteStream(LN_IN_buffer0, GeLU_OUT, PENUM, trans_en,GELU_MODE,TRANPOSE_MODE);
	}

}

void LayerNorm_SoftMax_GELU_TRANSPOSE_MUX(ap_uint<128> *input_shortcut_in,
	stream<ap_uint<DEQUAN_BIT*2> > LN_IN[MAX_OUP],
	stream<ap_uint<DEQUAN_BIT*2> > NO_LN_OUT[MAX_OUP],
    stream<ap_uint<ILN_OUT_WIDTH*2> > LN_OUT[MAX_NORM_PE],
	stream<ap_uint<DEQUAN_BIT*2> > NO_SOFTMAX_OUT[MAX_OUP], 
	stream<ap_uint<SOFTMAX_OUT_WIDTH*2> > SOFTMAX_OUT[MAX_SOFTMAX_STAGE2_PE],
	stream<ap_uint<GELU_OUT_WIDTH*2> > GELU_OUT[MAX_GELU_PE],
	const unsigned MM_M,
	const unsigned PENUM,
	const unsigned OUT_W,
	const unsigned OUT_H,
	const unsigned GROUPS,
	const bool SA_MODE,
	const bool NORM_MODE,
	const bool SOFTMAX_MODE,
	const bool EBMULT_MODE,
	const bool SHORCUT_ADD_MODE,
	const bool FCU_MODE,
	const bool GELU_MODE,
	const bool TRANSPOSE_MODE){
#pragma HLS INLINE OFF
	if(NORM_MODE==false&&SOFTMAX_MODE==false&&GELU_MODE==false&&TRANSPOSE_MODE==false&&EBMULT_MODE==false){
		return;
	}		

	if(NORM_MODE==true){
		LayerNorm(input_shortcut_in,LN_IN,NO_LN_OUT, LN_OUT,MM_M,PENUM,OUT_W, OUT_H, GROUPS,SA_MODE,NORM_MODE,SHORCUT_ADD_MODE,FCU_MODE);
	}
	else if(SOFTMAX_MODE==true||EBMULT_MODE==true){
		SOFTMAX_UNIT(LN_IN,NO_SOFTMAX_OUT, SOFTMAX_OUT,PENUM,GROUPS,SOFTMAX_MODE,EBMULT_MODE);
	}
	else if(GELU_MODE==true||TRANSPOSE_MODE==true){
		GeLU_TranPose_UNIT(LN_IN,GELU_OUT,PENUM,GROUPS,GELU_MODE,TRANSPOSE_MODE);
	}

}

void Write_Out_to_DDR_Shortcut(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
	 const unsigned PENUM,
    const unsigned OUT_W,
    const unsigned D,
    const unsigned OUT_H,
	const unsigned M_div_D,
	const unsigned NumLines,
	
	const unsigned WhichPath,
	const bool CONV1_TO_MM_EN,
	const bool skip_mode
	){

    if(skip_mode==false){
      return;
    }

	if(WhichPath==2&&CONV1_TO_MM_EN==true){
		Write_Shortcut_conv3_to_conv1(fifo_out,ddr_fm_result,OUT_W,D,OUT_H,M_div_D,NumLines);
	}
	else if(WhichPath==8){
		Write_to_DDR_NORM_MM_FM_SOFTMAX_GELU(fifo_out,ddr_fm_result,PENUM, M_div_D, NumLines);
	}
	else{
		Write_Out_to_DDR_Shortcut_DIRECT(fifo_out,ddr_fm_result,NumLines);
	}





}







void do_compute_kernel(
	            ap_uint<128>* img_mm, 
				ap_uint<128>* weight_conv3_mm,

				ap_uint<128>* ddr_fm_shortcut,

				ap_uint<128>* ddr_fm_back,
				ap_uint<128>* ddr_fm_shortcut_back,
				// stream<ap_uint<DEQUAN_BIT * 2> > fifo_C_deQua[MAX_OUP],
				const unsigned layer_weight_offset,
				const unsigned R,
				const unsigned C,
				const unsigned M,
				const unsigned N,
				const unsigned D,
				const unsigned WhichPath,
				const bool CONV1_TO_MM_EN,
				const bool SA_MODE,
				const bool NORM_MODE,
				const bool QUAN_MODE,
				const bool SHORCUT_QUAN_MODE,
				const bool SHORCUT_ADD_MODE,
				const bool FCU_MODE,
				const bool SOFTMAX_MODE,
				const bool EBMULT_MODE,
				const bool GELU_MODE,
				const bool TRANSPOSE_MODE){

#pragma HLS DATAFLOW


	unsigned num_a_in;  // MM or CONV3
	unsigned num_w_in;  // MM or CONV3
	unsigned num_a_sa;
	unsigned num_a_sa_out;
	unsigned num_w_sa;
	unsigned num_sa_res;
	unsigned num_out;
	unsigned num_out_softmax;
	unsigned num_norm_softmax_out;
	
	unsigned GROUPS;  // MM or CONV3
	unsigned PENUM;
	unsigned SIMDNUM;
	unsigned SA_NW;
	unsigned CONV_Inter_R=R+2;




	if(SA_MODE==false){  // MM

	    PENUM=M/(MAX_OUP*PACK_NUM);
		SIMDNUM=R/(MAX_INP*PACK_NUM);
		SA_NW=N;
		num_a_in=R*M*N/(MAX_INP*PACK_NUM*MAX_OUP*PACK_NUM);
		num_w_in=num_a_in;
		num_a_sa=num_a_in;
		num_a_sa_out=R*M/(MAX_A_ROW*MAX_A_COL*SA_OUP*PACK_OUT_NUM);
		num_w_sa=num_a_in;
		num_sa_res= R*M/(MAX_INP*MAX_OUP*PACK_OUT_NUM);
		num_out=R*M/(MAX_OUP*2);
		if(NORM_MODE){
			num_norm_softmax_out=R*PENUM*(MAX_OUP/MAX_NORM_PE);
			num_out_softmax=num_out;
		}
		else if(SOFTMAX_MODE){
			num_norm_softmax_out=(R*PENUM*(MAX_OUP/MAX_SOFTMAX_STAGE2_PE))>>1;
			
			num_out_softmax=num_out>>1;
		}
		else if(EBMULT_MODE){
			num_norm_softmax_out=num_out>>1;
			num_out_softmax=num_norm_softmax_out;
		}
		else if(GELU_MODE){
			num_norm_softmax_out=(R*PENUM*(MAX_OUP/MAX_SOFTMAX_STAGE2_PE))>>1;
			num_out_softmax=num_out>>1;

		}
		else if(TRANSPOSE_MODE){
			num_norm_softmax_out=(R*PENUM*(MAX_OUP/MAX_GELU_PE));
			num_out_softmax=num_out;
		}
		else{
			num_norm_softmax_out=num_out;
			num_out_softmax=num_out;
		}

		GROUPS=R/(MAX_INP*2);

	}
	else { // CONV

		GROUPS=M/D;
	    PENUM=D/MAX_OUP;
	    SIMDNUM=N/MAX_INP;
		SA_NW=C/2;
		num_a_in=R*C*N/(MAX_INP*PACK_NUM);
		
		num_a_sa=GROUPS*R*PENUM*(C/2)*CONV_K*SIMDNUM;
		num_a_sa_out=num_a_sa;
		num_w_in=((CONV_K*N)/MAX_INP)*(D/(MAX_A_COL));
		
		num_w_sa=GROUPS*R*PENUM*CONV_K*SIMDNUM*SA_OUP;
		
		num_sa_res=R*C*M/(MAX_OUP*2);
		num_out=R*C*M/(MAX_OUP*2);
		// scale_factor_buffer[0]=conv3_bias[scale_offset];
		num_norm_softmax_out=GROUPS*R*(PENUM)*(C/2)*(MAX_OUP/MAX_NORM_PE);
		num_out_softmax=num_out;

	}










#ifdef CONV3_BIAS_DEBUG

//	char fname_acc[100];
//	for(unsigned int c = 0; c < MAX_OUP; c++){
//		sprintf(fname_acc,"conv3_bias_l%d.txt",c);
//		FILE* fp_acc = fopen(fname_acc, "wb");
//		for (unsigned i = 0; i < CONV_GROUPS*R*(PENUM)*(C/2); i++) {
//			ap_int<BIAS_BIT> temp = to_conv_bias_in[c].read();
//			fprintf(fp_acc, "%d\n", int(temp));
//	  	}
//		fclose(fp_acc);
//	}

   	for (unsigned i = 0; i < CONV_GROUPS*R*(PENUM)*(C/2); i++) {
   #pragma HLS PIPELINE II=1
   	  for(unsigned int c = 0; c < MAX_OUP; c++){
   		ap_uint<BIAS_BIT> temp = to_conv_bias_in[c].read();
   	  }
   	}
#endif

#pragma region SA_code_Region

// A data path

	stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> > to_mm_a_in("to_mm_a_in");
#pragma HLS STREAM variable=to_mm_a_in depth=128

	stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> > to_conv3_a_in("to_conv3_a_in");
#pragma HLS STREAM variable=to_conv3_a_in depth=128

	ExtractPixels_AXI_MMA(img_mm,to_mm_a_in,to_conv3_a_in, N, SIMDNUM,PENUM, C, D, R, num_a_in,GROUPS,SA_MODE, WhichPath);       //NumLines=800*800/4


 stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> > conv3_samepad("conv3_samepad");
#pragma HLS STREAM variable=conv3_samepad depth=8

    SAMEPAD_DSPopt_SA_UP_DOWN<1,1,MAX_INP,PACK_NUM, IN_BIT>(to_conv3_a_in, conv3_samepad, CONV_Inter_R, C/2, N,GROUPS, SA_MODE);

stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> > conv3_sild("conv3_sild");
#pragma HLS STREAM variable=conv3_sild depth=4

	conv3padding_opt_SA<3, IN_BIT, MAX_INP>(conv3_samepad, conv3_sild,CONV_Inter_R,C,R, N, PENUM,GROUPS, SA_MODE);



stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> > to_sa_a_in("to_sa_a_in");
#pragma HLS STREAM variable=to_sa_a_in depth=4

	MuxStream2<MAX_INP * IN_BIT * PACK_NUM>(to_mm_a_in,conv3_sild,to_sa_a_in,SA_MODE, num_a_sa);


stream<ap_uint< SA_INP* IN_BIT * PACK_NUM> > fifo_SA_A_PE[MAX_A_ROW][MAX_A_COL];  //SIMD PE
#pragma HLS STREAM variable=fifo_SA_A_PE depth=4


    A_to_array<MAX_A_ROW,MAX_A_COL,MAX_INP * IN_BIT * PACK_NUM,SA_INP * IN_BIT * PACK_NUM>(
        to_sa_a_in,
        fifo_SA_A_PE,
        num_a_sa
    );


// W data path

	stream<ap_uint<MAX_OUP * W_BIT * PACK_NUM> > mm_w_stream_extract("mm_w_stream_extract");
#pragma HLS STREAM variable=mm_w_stream_extract depth=128
stream<ap_uint<MAX_INP * CONV_K *W_BIT> > fifo_CONV_W_in[MAX_A_COL];
#pragma HLS STREAM variable=fifo_CONV_W_PE depth=128 dim=1

	ExtractPixels_AXI_MMW_CONV(weight_conv3_mm,mm_w_stream_extract,fifo_CONV_W_in, N, SIMDNUM,PENUM, num_w_in,R,GROUPS,SA_MODE,layer_weight_offset);       //MM




	stream<ap_uint<MAX_OUP * W_BIT * PACK_CONV_NUM> > mm_w_stream_exp("mm_w_stream_exp");
#pragma HLS STREAM variable=mm_w_stream_exp depth=4

	MM_to_CONV3_Stream<MAX_OUP,PACK_NUM,PACK_CONV_NUM,W_BIT>(mm_w_stream_extract,mm_w_stream_exp,num_w_in,SA_MODE);



stream<ap_uint<SA_OUP * CONV_K * W_BIT> > fifo_MM_W_PE[MAX_A_ROW][MAX_A_COL];
#pragma HLS STREAM variable=fifo_MM_W_PE depth=4 dim=1
#pragma HLS STREAM variable=fifo_MM_W_PE depth=4 dim=2

    W_mm_to_array<MAX_A_ROW,MAX_A_COL,MAX_OUP * W_BIT * PACK_CONV_NUM,SA_OUP * W_BIT * PACK_CONV_NUM>(
        mm_w_stream_exp,
        fifo_MM_W_PE,
        num_w_in,
		SA_MODE
    );




stream<ap_uint<SA_INP * CONV_K * W_BIT> > fifo_CONV_W_PE[MAX_A_ROW][MAX_A_COL];
#pragma HLS STREAM variable=fifo_CONV_W_PE depth=8 dim=1
#pragma HLS STREAM variable=fifo_CONV_W_PE depth=8 dim=2

	W_conv3_array<MAX_A_ROW,MAX_A_COL, MAX_OUP,SA_OUP, MAX_INP,SA_INP,
	CONV_K,MAX_CONV3_WEIGHT_LENGTH,W_BIT>
	(fifo_CONV_W_in, fifo_CONV_W_PE, R,num_w_in, GROUPS ,SA_MODE);


stream<ap_uint<SA_OUP * CONV_K * W_BIT> > fifo_SA_W_PE[MAX_A_ROW][MAX_A_COL];
#pragma HLS STREAM variable=fifo_SA_W_PE depth=8 dim=1
#pragma HLS STREAM variable=fifo_SA_W_PE depth=8 dim=2

	MuxStream2_RC<MAX_A_ROW,MAX_A_COL,SA_OUP * CONV_K * W_BIT>(fifo_MM_W_PE,fifo_CONV_W_PE,fifo_SA_W_PE, num_w_sa,SA_MODE);




 stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > fifo_SA_O_PE[MAX_A_ROW][MAX_A_COL][SA_OUP]; // [SIMD][PE]
 #pragma HLS BIND_STORAGE variable=fifo_SA_O_PE type=fifo impl=autosrl
 #pragma HLS STREAM variable=fifo_SA_O_PE depth=16 dim=1
 #pragma HLS STREAM variable=fifo_SA_O_PE depth=16 dim=2
 #pragma HLS STREAM variable=fifo_SA_O_PE depth=16 dim=3


 for (unsigned int r=0; r< MAX_A_ROW;r++){  //有问题
 #pragma HLS UNROLL
     for (unsigned int c=0; c< MAX_A_COL;c++){
     #pragma HLS UNROLL

			PE_wrapper<IN_BIT,W_BIT,PACK_NUM,PACK_CONV_NUM,PACK_OUT_NUM,SA_INP,
					SA_OUP,ACC_BIT>(
						r,
						c,
						fifo_SA_A_PE[r][c],
						fifo_SA_W_PE[r][c],
						fifo_SA_O_PE[r][c],
						SA_NW,
						num_a_sa,
						SA_MODE);

			#ifdef PRINT_DEBUG
				cout<<"PE_wrapper_"<<r<<"_"<<c<<" fininshed!"<<endl;
			#endif

			}
		}


//	for (unsigned i = 0; i < num_a_sa; i++) {
//#pragma HLS PIPELINE II=1
//	  for(unsigned int c = 0; c < MAX_A_COL; c++){
//		for(unsigned int m = 0; m < SA_OUP; m++){
//		  for(unsigned int r = 0; r < MAX_A_ROW; r++){
//			ap_uint<ACC_BIT * PACK_OUT_NUM> temp = fifo_SA_O_PE[r][c][m].read();
//		  }
//		}
//	  }
//	}

#ifdef DEBUG
	ap_uint<ACC_BIT * PACK_OUT_NUM> res_tmp;
	char fp_name[100];
	int test_R=128;
	int test_M=128;
	for (unsigned int c=0; c< MAX_A_COL;c++){
		for(unsigned int y=0; y< SA_INP;y++){
			int index=c*SA_INP+y;
			sprintf(fp_name,"mm_output_pe%d.txt",index);
			FILE* fp1 = fopen(fp_name, "wb");
			for(int lines=0;lines<test_R*test_M/(MAX_INP*MAX_OUP*4);lines++){
				for (unsigned int r=0; r< MAX_A_ROW;r++){  //有问题
					for(unsigned int x=0; x< SA_INP;x++){
						res_tmp=fifo_SA_O_PE[r][c][y].read();
						fifo_SA_O_PE[r][c][y].write(res_tmp);
						for(int m=0; m<PACK_OUT_NUM;m++){
							ap_int<ACC_BIT> tmp=res_tmp((m+1)*ACC_BIT-1,m*ACC_BIT);
							fprintf(fp1, "%d\n", (int)tmp);
						}

					}
				}
			}
			fclose(fp1);
		}
	}
#endif




stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > fifo_C_MM[MAX_A_ROW][MAX_A_COL][SA_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_MM type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_MM depth=8 dim=1
#pragma HLS STREAM variable=fifo_C_MM depth=8 dim=2
#pragma HLS STREAM variable=fifo_C_MM depth=8 dim=3

stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > fifo_C_CONV3[MAX_A_ROW][MAX_A_COL][SA_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_CONV3 type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_CONV3 depth=4 dim=1
#pragma HLS STREAM variable=fifo_C_CONV3 depth=4 dim=2
#pragma HLS STREAM variable=fifo_C_CONV3 depth=4 dim=3


	DemuxStream2_ARRAY(fifo_SA_O_PE,fifo_C_MM,fifo_C_CONV3,num_a_sa_out,SA_MODE);



stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > fifo_C_MM_RES[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_MM_RES type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_MM_RES depth=16 dim=1

	MM_Parallel_to_Serial_Out<MAX_A_ROW,MAX_A_COL,MAX_OUP,SA_OUP,PACK_OUT_NUM,ACC_BIT>(
				fifo_C_MM,
				fifo_C_MM_RES,
				num_sa_res*SA_INP,
				SA_MODE);

stream<ap_uint<ACC_BIT * 2> > fifo_C_MM_RES_Redu[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_MM_RES_REDU type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_MM_RES_REDU depth=4 dim=1

	ReduceWidth_P<ACC_BIT * PACK_OUT_NUM ,ACC_BIT * 2,MAX_OUP >(fifo_C_MM_RES,fifo_C_MM_RES_Redu,num_out/2,SA_MODE);



#ifdef MM_OS_RED_DEBUG
	ap_uint<ACC_BIT * 2> data;

    for(int i = 0;i < num_out;i++){
	#pragma HLS PIPELINE II=1
        for(int j = 0;j < MAX_OUP;j++){
        	data=fifo_C_MM_RES_Redu[j].read();
        }
    }
#endif

stream<ap_uint<ACC_BIT * PACK_OUT_NUM> > fifo_C_CONV3_ACC[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_CONV3_ACC type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_CONV3_ACC depth=4 dim=1


	arrar_acc_to_Res<MAX_A_ROW,MAX_A_COL,MAX_OUP,SA_OUP,PACK_OUT_NUM,ACC_BIT>(fifo_C_CONV3,fifo_C_CONV3_ACC,num_a_sa,SA_MODE);






stream<ap_uint<ACC_BIT * 2> > fifo_C_CONV3_INTER[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_CONV3_INTER type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_CONV3_INTER depth=4 dim=1

	PE_DSP_ACC<CONV_K,MAX_OUP,ACC_BIT>(fifo_C_CONV3_ACC,fifo_C_CONV3_INTER, R,C,PENUM,SIMDNUM,GROUPS,SA_MODE);


stream<ap_uint<ACC_BIT * 2> > fifo_C_CONV3_RES[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_CONV3_RES type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_CONV3_RES depth=16 dim=1

	Inter_Reorg_acc_to_Res<CONV_K, ACC_BIT, MAX_OUP>(fifo_C_CONV3_INTER,fifo_C_CONV3_RES, R, C, PENUM,SIMDNUM,GROUPS, SA_MODE);



   stream<ap_uint<ACC_BIT * 2> > fifo_C_RES[MAX_OUP]; // [SIMD][PE]
   #pragma HLS BIND_STORAGE variable=fifo_C_RES type=fifo impl=autosrl
   #pragma HLS STREAM variable=fifo_C_RES depth=4 dim=1

	MuxStream2_P<ACC_BIT * 2, MAX_OUP>(fifo_C_MM_RES_Redu,fifo_C_CONV3_RES,fifo_C_RES, num_out, SA_MODE);

#pragma endregion

#ifdef DEBUG

	FILE* fp1_SA_OUT = fopen("result_SA_OUT.txt", "wb");

	ap_uint<ACC_BIT * 2> tempin;
	ap_int<ACC_BIT> temp0,temp1;

	for(int i=0; i<num_out;i++){
		for(int j=0; j<MAX_OUP;j++){
			tempin=fifo_C_RES[j].read();
			fifo_C_RES[j].write(tempin);
			// cout<<tempin<<endl;
			
			(temp1,temp0)=tempin;

		// 	cout<<"out0:" <<temp1<<endl;
		// 	cout<<"out1:" <<temp0<<endl;

        //   if(temp0!=0||temp1!=0){
        //     cout<<"debug"<<endl;
        //   }

			fprintf(fp1_SA_OUT, "%lf\n", double(temp0));
			fprintf(fp1_SA_OUT, "%lf\n", double(temp1));
		}
	}

	fclose(fp1_SA_OUT);
#endif

stream<ap_uint<DEQUAN_BIT * 2> > fifo_C_deQua[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_deQua type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_deQua depth=4 dim=1

	Dequan_to_Res(fifo_C_RES,fifo_C_deQua, PENUM,SIMDNUM, C, R,GROUPS, num_out,SA_MODE);


#ifdef DEBUG


	FILE* fp1_dequan = fopen("result_SA_OUT_DEQUAN.txt", "wb");

	ap_uint<DEQUAN_BIT * 2> Dequan_to_Res_tempin;
	ap_uint<DEQUAN_BIT> Dequan_out0, Dequan_out1;
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> Dequan_res_out0,Dequan_res_out1;

	for(int i=0; i<num_out;i++){
		for(int j=0; j<MAX_OUP;j++){
			Dequan_to_Res_tempin=fifo_C_deQua[j].read();
			fifo_C_deQua[j].write(Dequan_to_Res_tempin);
			// cout<<Dequan_to_Res_tempin<<endl;
			(Dequan_out1,Dequan_out0)=Dequan_to_Res_tempin;
			Dequan_res_out0(DEQUAN_BIT-1,0)=Dequan_out0(DEQUAN_BIT-1,0);
			Dequan_res_out1(DEQUAN_BIT-1,0)=Dequan_out1(DEQUAN_BIT-1,0);

			// cout<<Dequan_res_out0<<endl;
			// cout<<Dequan_res_out1<<endl;
			fprintf(fp1_dequan, "%lf\n", double(Dequan_res_out0));
			fprintf(fp1_dequan, "%lf\n", double(Dequan_res_out1));
		}
	}

	fclose(fp1_dequan);
#endif



stream<ap_uint<DEQUAN_BIT * 2> > fifo_shortcut_Qua[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_shortcut_Qua type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_shortcut_Qua depth=4 dim=1


stream<ap_uint<DEQUAN_BIT * 2> > fifo_norm_in[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_norm_in type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_norm_in depth=4 dim=1



	DuplicateStreamN_OUP(fifo_C_deQua,fifo_shortcut_Qua,fifo_norm_in, num_out,
	                     NORM_MODE, SHORCUT_QUAN_MODE,SOFTMAX_MODE,EBMULT_MODE, GELU_MODE,TRANSPOSE_MODE,SHORCUT_ADD_MODE);




// #ifdef CONV_WS_DEBUG
// 	ap_uint<SILU_BIT * 2> data_qua_temp;

//     for(int i = 0;i < num_out;i++){
// 	#pragma HLS PIPELINE II=1
//         for(int j = 0;j < MAX_OUP;j++){
//         	data_qua_temp=fifo_shortcut_out[j].read();
//         }
//     }
// #endif




stream<ap_uint<ILN_OUT_WIDTH * 2> > fifo_norm_out[MAX_NORM_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_norm_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_norm_out depth=4 dim=1

stream<ap_uint<DEQUAN_BIT * 2> > fifo_no_norm_out[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_no_norm_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_no_norm_out depth=4 dim=1

stream<ap_uint<SOFTMAX_OUT_WIDTH * 2> > fifo_softmax_out[MAX_SOFTMAX_STAGE2_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_softmax_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_softmax_out depth=4 dim=1

stream<ap_uint<DEQUAN_BIT * 2> > fifo_no_emulti_out[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_no_emulti_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_no_emulti_out depth=4 dim=1


stream<ap_uint<GELU_OUT_WIDTH * 2> > fifo_gelu_tranpose_out[MAX_GELU_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_gelu_tranpose_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_gelu_tranpose_out depth=4 dim=1




	LayerNorm_SoftMax_GELU_TRANSPOSE_MUX(ddr_fm_shortcut,fifo_norm_in,fifo_no_norm_out,fifo_norm_out,fifo_no_emulti_out, fifo_softmax_out,
						fifo_gelu_tranpose_out,
	                    M,PENUM,C, R, GROUPS,
	                     SA_MODE,NORM_MODE,SOFTMAX_MODE,EBMULT_MODE,SHORCUT_ADD_MODE,FCU_MODE, GELU_MODE,TRANSPOSE_MODE);




#ifdef TEST_DEBUG


	FILE* fp1_norm = fopen("result_norm_out.txt", "wb");

	ap_uint<ILN_OUT_WIDTH * 2> norm_tempin;
	ap_uint<ILN_OUT_WIDTH> norm_out0, norm_out1;

	for(int i=0; i<num_norm_softmax_out;i++){
		for(int j=0; j<MAX_NORM_PE;j++){
			norm_tempin=fifo_norm_out[j].read();
			fifo_norm_out[j].write(norm_tempin);
			// cout<<Dequan_to_Res_tempin<<endl;
			(norm_out1,norm_out0)=norm_tempin;


			// cout<<Dequan_res_out0<<endl;
			// cout<<Dequan_res_out1<<endl;
			fprintf(fp1_norm, "%lf\n", double(norm_out0));
			fprintf(fp1_norm, "%lf\n", double(norm_out1));
		}
	}

	fclose(fp1_norm);
#endif


stream<ap_uint<SILU_BIT * 2> > fifo_silu_out[MAX_NORM_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_silu_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_silu_out depth=4 dim=1

stream<ap_uint<IN_BIT * 2> > fifo_silu_quan_out[MAX_NORM_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_silu_quan_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_silu_quan_out depth=4 dim=1
	SiLU_Quan_Unit(fifo_norm_out,fifo_silu_quan_out,num_norm_softmax_out,NORM_MODE,SA_MODE);


#ifdef TEST_DEBUG


	FILE* fp1_silu = fopen("result_silu_out.txt", "wb");

	ap_uint<IN_BIT * 2> silu_tempin;
	ap_uint<IN_BIT> silu_out0, silu_out1;

	for(int i=0; i<num_norm_softmax_out;i++){
		for(int j=0; j<MAX_NORM_PE;j++){
			silu_tempin=fifo_silu_quan_out[j].read();
			fifo_silu_quan_out[j].write(silu_tempin);
			// cout<<Dequan_to_Res_tempin<<endl;
			(silu_out1,silu_out0)=silu_tempin;


			// cout<<Dequan_res_out0<<endl;
			// cout<<Dequan_res_out1<<endl;
			fprintf(fp1_silu, "%lf\n", double(silu_out0));
			fprintf(fp1_silu, "%lf\n", double(silu_out1));
		}
	}

	fclose(fp1_silu);
#endif



stream<ap_uint<IN_BIT * 2> > fifo_softmax_quan_out[MAX_SOFTMAX_STAGE2_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_softmax_quan_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_softmax_quan_out depth=4 dim=1


	Softmax_Quan_Unit(fifo_softmax_out,fifo_softmax_quan_out,num_norm_softmax_out,SOFTMAX_MODE);




stream<ap_uint<IN_BIT * 2> > fifo_gelu_tranpose_quan_out[MAX_GELU_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_gelu_tranpose_quan_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_gelu_tranpose_quan_out depth=4 dim=1

	Gelu_Quan_Unit(fifo_gelu_tranpose_out,fifo_gelu_tranpose_quan_out,num_norm_softmax_out,GELU_MODE|TRANSPOSE_MODE);




stream<ap_uint<IN_BIT * 2> > fifo_quan_in[MAX_NORM_PE]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_quan_in type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_quan_in depth=4 dim=1


	MuxStream3_P<IN_BIT * 2,MAX_NORM_PE >(fifo_silu_quan_out, fifo_softmax_quan_out,fifo_gelu_tranpose_quan_out,fifo_quan_in,num_norm_softmax_out,
	           NORM_MODE,SOFTMAX_MODE,GELU_MODE,TRANSPOSE_MODE);


stream<ap_uint<IN_BIT * 2> > fifo_quan_out[MAX_OUP]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_quan_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_quan_out depth=4 dim=1

	ExpandWidth_OUP<IN_BIT * 2,MAX_NORM_PE,MAX_OUP>(fifo_quan_in,fifo_quan_out,num_norm_softmax_out,QUAN_MODE);




	stream<ap_uint<IN_BIT * 2* MAX_OUP> > fifo_quan_results; // [SIMD][PE]
	#pragma HLS BIND_STORAGE variable=fifo_quan_results type=fifo impl=autosrl
	#pragma HLS STREAM variable=fifo_quan_results depth=4 dim=1

	Nonlinear_QuanUnit(fifo_quan_out,fifo_quan_results,num_out_softmax, QUAN_MODE);



	stream<ap_uint<DEQUAN_BIT * 2> > fifo_shortcut_in[MAX_OUP]; // [SIMD][PE]
	#pragma HLS BIND_STORAGE variable=fifo_shortcut_in type=fifo impl=autosrl
	#pragma HLS STREAM variable=fifo_shortcut_in depth=4 dim=1

	MuxStream3_P_BRANCH<DEQUAN_BIT * 2,MAX_OUP >(fifo_shortcut_Qua, fifo_no_norm_out , fifo_no_emulti_out, fifo_shortcut_in,num_out_softmax,
	   SHORCUT_ADD_MODE,EBMULT_MODE, SHORCUT_QUAN_MODE);

stream<ap_uint<IN_BIT * 2*MAX_OUP> > fifo_shortcut_out; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_shortcut_out type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_shortcut_out depth=4 dim=1


	ShortcutQuanUnit(fifo_shortcut_in,fifo_shortcut_out, num_out_softmax,SHORCUT_QUAN_MODE,SHORCUT_ADD_MODE,FCU_MODE);

stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > fifo_quan_results_red; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_quan_results_red type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_quan_results_red depth=128 dim=1


	ReduceWidth_EN<IN_BIT * 2*MAX_OUP,IN_BIT * 2*MAX_OUP/2>(fifo_quan_results,fifo_quan_results_red,num_out_softmax,QUAN_MODE);


	Write_Out_to_DDR_NORM(fifo_quan_results_red,ddr_fm_back,PENUM,  C, D, R,  GROUPS, num_out_softmax*2, WhichPath, QUAN_MODE);



stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > fifo_shortcut_out_red; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_shortcut_out_red type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_shortcut_out_red depth=128 dim=1


	ReduceWidth_EN<IN_BIT * 2*MAX_OUP,IN_BIT * 2*MAX_OUP/2>(fifo_shortcut_out,fifo_shortcut_out_red,num_out_softmax,SHORCUT_QUAN_MODE);

	Write_Out_to_DDR_Shortcut(fifo_shortcut_out_red, ddr_fm_shortcut_back,PENUM,C,D,R,GROUPS ,num_out_softmax*2,  WhichPath, CONV1_TO_MM_EN  ,SHORCUT_QUAN_MODE);



}


void do_compute_top(ap_uint<128>* img_conv3_mm,
				// conv3的权重输入
				ap_uint<128> *weight_conv3_mm, 

				ap_uint<128>* ddr_bias_scale_factor,  //  BIAS_BIT*16

				ap_uint<128>* ddr_fm_shortcut,

				ap_uint<128>* ddr_fm_back,
				ap_uint<128>* ddr_fm_shortcut_back,
				// stream<ap_uint<DEQUAN_BIT * 2> > fifo_C_deQua[MAX_OUP],

				const unsigned layer_bias_offset,
				const unsigned layer_weight_offset,
				// const ap_uint<4> ENCODE_MODE,
				const unsigned R,
				const unsigned C,
				const unsigned N,
				const unsigned M,
				const unsigned D,
				const unsigned WhichPath,
				const bool CONV1_TO_MM_EN
				){

#pragma HLS INTERFACE m_axi depth=2560 bundle=img_fm_bus port=img_conv3_mm offset=slave
#pragma HLS INTERFACE m_axi depth=2560 bundle=weight_bus port=weight_conv3_mm offset=slave

// 合并为一个端口
#pragma HLS INTERFACE m_axi depth=1280 bundle=other_bus port=ddr_fm_shortcut offset=slave

#pragma HLS INTERFACE m_axi depth=33 bundle=other_bus port=ddr_bias_scale_factor offset=slave

#pragma HLS INTERFACE m_axi depth=2560 bundle=ddr_fm_back_bus port=ddr_fm_back offset=slave
#pragma HLS INTERFACE m_axi depth=0 bundle=ddr_fm_shortcut_back_bus port=ddr_fm_shortcut_back offset=slave

#pragma HLS INTERFACE s_axilite port=layer_bias_offset bundle=control
#pragma HLS INTERFACE s_axilite port=layer_weight_offset bundle=control
#pragma HLS INTERFACE s_axilite port=R bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=N bundle=control
#pragma HLS INTERFACE s_axilite port=M bundle=control
#pragma HLS INTERFACE s_axilite port=D bundle=control
#pragma HLS INTERFACE s_axilite port=WhichPath bundle=control
#pragma HLS INTERFACE s_axilite port=CONV1_TO_MM_EN bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control




#pragma HLS ARRAY_PARTITION variable=conv3_w_buffer1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=conv3_w_buffer0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=conv3_mm_bias_buffer0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=conv3_mm_bias_buffer1 dim=1 complete

//#pragma HLS ARRAY_PARTITION variable=scale_factor dim=1 complete
#pragma HLS ARRAY_PARTITION variable=conv3_mm_bias_buffer dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=conv3_mm_bias_buffer1 dim=1 complete


#pragma HLS ARRAY_PARTITION variable=LN_IN_buffer0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=LN_IN_buffer1 dim=1 complete

#pragma HLS ARRAY_PARTITION variable=ln_ptf_factor_buffer0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=ln_ptf_factor_buffer1 dim=1 complete
// #pragma HLS ARRAY_PARTITION variable=LN_IN_buffer0 dim=1 complete
// #pragma HLS ARRAY_PARTITION variable=LN_IN_buffer1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=ln_gamma_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=ln_beta_buffer dim=1 complete

//#pragma HLS ARRAY_PARTITION variable=scale_factor_buffer dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=linear1d_weight dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=linear1d_bias_out_buffer dim=1 complete
//#pragma HLS ARRAY_PARTITION variable=linear1d_out_buffer dim=1 complete


	bool SA_MODE;
	bool NORM_MODE;
	bool QUAN_MODE;
	bool SHORTCUT_QUAN_MODE;
	bool FCU_MODE;
	bool SHORTCUT_ADD_MODE;
	bool SOFTMAX_MODE;
	bool EBMULT_MODE;
	bool GELU_MODE;
	bool TRANSPOSE_MODE;



	if(WhichPath==0){
		SA_MODE=1;
		NORM_MODE=1;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=1; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==1){
		SA_MODE=1;
		NORM_MODE=1;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=0; 
		FCU_MODE=1;
		SHORTCUT_ADD_MODE=1;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==2){
		SA_MODE=1;
		NORM_MODE=1;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=1; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=1;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==3){
		SA_MODE=0;
		NORM_MODE=0;
		QUAN_MODE=0;
		SHORTCUT_QUAN_MODE=1; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}

	else if(WhichPath==4){
		SA_MODE=0;
		NORM_MODE=1;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=1; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==5){
		SA_MODE=0;
		NORM_MODE=0;
		QUAN_MODE=0;
		SHORTCUT_QUAN_MODE=1; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==6){
		SA_MODE=0;
		NORM_MODE=0;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=0; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=1;
	}
	else if(WhichPath==7){
		SA_MODE=0;
		NORM_MODE=0;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=0; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=1;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==8){
		SA_MODE=0;
		NORM_MODE=0;
		QUAN_MODE=0;
		SHORTCUT_QUAN_MODE=1; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=1;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==9){
		SA_MODE=0;
		NORM_MODE=1;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=1; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=1;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==10){
		SA_MODE=0;
		NORM_MODE=0;
		QUAN_MODE=1;
		SHORTCUT_QUAN_MODE=0; 
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=1;
		TRANSPOSE_MODE=0;
	}
	else if(WhichPath==11){
		SA_MODE=1;
		NORM_MODE=0;
		QUAN_MODE=0;
		SHORTCUT_QUAN_MODE=1;
		FCU_MODE=0;
		SHORTCUT_ADD_MODE=0;
		SOFTMAX_MODE=0;
		EBMULT_MODE=0;
		GELU_MODE=0;
		TRANSPOSE_MODE=0;
	}

	ExtractPixels_AXI_AllBias(ddr_bias_scale_factor, M,D,layer_bias_offset,SA_MODE,NORM_MODE);


    do_compute_kernel(img_conv3_mm,weight_conv3_mm,ddr_fm_shortcut, ddr_fm_back,ddr_fm_shortcut_back,layer_weight_offset,
					R, C, M, N, D,WhichPath,CONV1_TO_MM_EN,
					SA_MODE, NORM_MODE,QUAN_MODE, SHORTCUT_QUAN_MODE,SHORTCUT_ADD_MODE, FCU_MODE,SOFTMAX_MODE,
					EBMULT_MODE, GELU_MODE,TRANSPOSE_MODE);
 


}

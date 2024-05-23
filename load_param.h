#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const

#include "config.h"
#include <ap_int.h>
using namespace hls;
using namespace std;

// #define OUTPUTW_DEBUG


void WriteMMFMParam_MMTRANSFER(ap_uint<128>* in, ap_uint< MAX_INP * IN_BIT * PACK_NUM> mm_a_buf[MAX_MM_FM_LENGTH],
	const unsigned MM_N){
#pragma HLS INLINE OFF




    ap_uint<IN_BIT * 2*MAX_OUP/2> BUFA_80b[MAX_INP];
    ap_uint<IN_BIT * 2*MAX_OUP/2> BUFB_80b[MAX_INP];
    #pragma HLS ARRAY_PARTITION variable=BUFA_80b complete dim=0
    #pragma HLS ARRAY_PARTITION variable=BUFB_80b complete dim=0

    ap_uint<128> temp_128b;
    ap_uint<80> temp_80b;
    ap_uint< MAX_INP * IN_BIT * PACK_NUM > temp_oup; 

    bool arb = 0;
    unsigned int max_inp_cnt=0;
    unsigned int cnt_write=0;

	for(unsigned i=0; i<MM_N*2*2;i++){ // 2个MAX_OUP,每个OUP拆成两个
	#pragma HLS PIPELINE II=1

        temp_128b = in[i];
        temp_80b=temp_128b;

        if(arb==0){
            BUFA_80b[max_inp_cnt]=temp_80b;
        }
        else{
            BUFB_80b[max_inp_cnt]=temp_80b;
        }
        

        if(i>=MAX_INP&&max_inp_cnt<(MAX_OUP/2)){
            for(unsigned m=0;m<MAX_INP;m++){
#pragma HLS UNROLL
                if(arb==0){
                    temp_80b=BUFB_80b[m]((max_inp_cnt+1)*IN_BIT * 2-1,max_inp_cnt*IN_BIT * 2);
                }
                else{
                    temp_80b=BUFA_80b[m]((max_inp_cnt+1)*IN_BIT * 2-1,max_inp_cnt*IN_BIT * 2);
                }
                temp_oup((m+1)*IN_BIT * 2-1,m*IN_BIT * 2)=temp_80b;
            }
            mm_a_buf[cnt_write]=temp_oup;
            cnt_write++;
        }


        if(max_inp_cnt==MAX_INP-1){
            max_inp_cnt=0;
            arb = !arb;
        }
        else{
            max_inp_cnt++;
        }

    }


	for(unsigned i=0; i<MAX_OUP/2;i++){ // 2个MAX_OUP,每个OUP拆成两个
	#pragma HLS PIPELINE II=1
        for(unsigned m=0;m<MAX_INP;m++){
    #pragma HLS UNROLL
            if(arb==0){
                temp_80b=BUFA_80b[m]((i+1)*IN_BIT * 2-1,i*IN_BIT * 2);
            }
            else{
                temp_80b=BUFB_80b[m]((i+1)*IN_BIT * 2-1,i*IN_BIT * 2);
            }
            temp_oup((m+1)*IN_BIT * 2-1,m*IN_BIT * 2)=temp_80b;
        }
        mm_a_buf[cnt_write]=temp_oup;
        cnt_write++;
    }


}


void WriteMMFMParam_DIRECT(ap_uint<128>* in, ap_uint< MAX_INP * IN_BIT * PACK_NUM> mm_a_buf[MAX_MM_FM_LENGTH],
	const unsigned MM_N){
#pragma HLS INLINE OFF

	ap_uint<128> weight_in[3];
	ap_uint<384> weight_in_384b;

	unsigned int bitIdx=0;
	unsigned int colIdx=0;
	ap_uint< MAX_INP * IN_BIT * PACK_NUM > temp;  // 20*8*2=320b
	for(unsigned i=0; i<MM_N*3;i++){
	#pragma HLS PIPELINE II=1
		weight_in[bitIdx] = in[i];
		// cout <<"The Value of Var_a: \t" <<temp<< " \t Binary format: \t" <<temp.to_string(2).c_str()<< '\n';
        if(bitIdx==3-1){
			weight_in_384b=(weight_in[2],weight_in[1],weight_in[0]);
			temp=weight_in_384b;
            mm_a_buf[colIdx]=temp;
        }
		

        if(bitIdx==3-1){
            bitIdx=0;
            if(colIdx==MAX_MM_FM_LENGTH-1){
                colIdx=0;
            }
            else{
                colIdx++;
            }
        }
        else{
            bitIdx++;
        }

	}
}


void Write_Out_to_DDR_Shortcut_DIRECT(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
	const unsigned NumLines){


	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;
	ap_uint<128> temp_128b;


        for (unsigned rep = 0; rep < NumLines; rep++) { 
    #pragma HLS PIPELINE II=1
            temp=fifo_out.read();
            temp_128b=temp;
            ddr_fm_result[rep]=temp_128b;

        }
}


void Write_Shortcut_conv3_to_conv1(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
    const unsigned OUT_W,
    const unsigned D,
    const unsigned OUT_H,
	const unsigned M_div_D,
	const unsigned NumLines){

    unsigned int loop0,loop1,loop2,loop3,loop4;

    unsigned int loop0_cnt=0;
    unsigned int loop1_cnt=0;
    unsigned int loop2_cnt=0;
    unsigned int loop3_cnt=0;
    unsigned int loop4_cnt=0;
    // unsigned int loop5_cnt=0;


	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;
	ap_uint<128> temp_128b;
    unsigned index;


    loop0=2;  // MAX_OUP本身拆成的2个
    loop1=OUT_W/2;
    loop2=OUT_H;
    loop3=D/MAX_OUP;
    loop4=M_div_D;

	for (unsigned rep = 0; rep < NumLines; rep++) { 
#pragma HLS PIPELINE II=1
		temp=fifo_out.read();
		temp_128b=temp;

        // 存数按照数据下次用的顺序
        index=loop4_cnt*loop3*loop2*loop1*loop0+loop3_cnt*loop2*loop1*loop0+loop2_cnt*loop1*loop0+loop1_cnt*loop0+loop0_cnt;
		
        ddr_fm_result[rep]=temp_128b;


        // 取数按照数据来的顺序
        if(loop0_cnt==loop0-1){
            loop0_cnt=0;
            if(loop1_cnt==loop1-1){
                loop1_cnt=0;
                if(loop3_cnt==loop3-1){
                    loop3_cnt=0;
                    if(loop2_cnt==loop2-1){
                        loop2_cnt=0;
                        if(loop4_cnt==loop4-1){
                            loop4_cnt=0;
                        }
                        else{
                            loop4_cnt++;
                        }
                    }
                    else{
                        loop2_cnt++;
                    }
                }
                else{
                    loop3_cnt++;
                }
            }
            else{
                loop1_cnt++;
            }
        }
        else{
            loop0_cnt++;
        }
	}

}

void Write_Out_to_DDR_DIRECT(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
	const unsigned NumLines
	){



	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;
	ap_uint<128> temp_128b;


	for (unsigned rep = 0; rep < NumLines; rep++) { 
#pragma HLS PIPELINE II=1
		temp=fifo_out.read();
		temp_128b=temp;
		ddr_fm_result[rep]=temp_128b;

	}


}



void Write_to_DDR_NORM_MM_FM_SOFTMAX_GELU(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
    const unsigned PENUM,
    const unsigned R_div_2INP,  // Groups
	const unsigned NumLines){


    unsigned int loop0,loop1,loop2,loop3;

    unsigned int loop0_cnt=0;
    unsigned int loop1_cnt=0;
    unsigned int loop2_cnt=0;
    unsigned int loop3_cnt=0;

    loop0=MAX_INP;  // MAX_OUP本身拆成的2个
    loop1=4;   // 除了MAXOUP是2个， 还有MAX_OUP拆成两个
    loop2=PENUM/2;
    loop3=R_div_2INP; 





	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;
	ap_uint<128> temp_128b;
    unsigned index;


	for (unsigned rep = 0; rep < NumLines; rep++) { 
#pragma HLS PIPELINE II=1
		temp=fifo_out.read();
		temp_128b=temp;

        // 存数按照数据下次用的顺序
        index=loop3_cnt*loop2*loop1*loop0+loop2_cnt*loop1*loop0+loop1_cnt*loop0+loop0_cnt;
		
        ddr_fm_result[rep]=temp_128b;


        // 取数按照数据来的顺序
        if(loop1_cnt==loop1-1){
            loop1_cnt=0;
            if(loop2_cnt==loop2-1){
                loop2_cnt=0;
                if(loop0_cnt==loop0-1){
                    loop0_cnt=0;
                    if(loop3_cnt==loop3-1){
                        loop3_cnt=0;
                    }
                    else{
                        loop3_cnt++;
                    }
                }
                else{
                    loop0_cnt++;
                }
            }
            else{
                loop2_cnt++;
            }
        }
        else{
            loop1_cnt++;
        }
	}
    
}



void Write_to_DDR_NORM_MM_FM(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
    const unsigned PENUM,
    const unsigned R_div_2INP,  // Groups
	const unsigned NumLines){


    unsigned int loop0,loop1,loop2,loop3;

    unsigned int loop0_cnt=0;
    unsigned int loop1_cnt=0;
    unsigned int loop2_cnt=0;
    unsigned int loop3_cnt=0;

    loop0=MAX_INP;  // MAX_OUP本身拆成的2个
    loop1=4;   // 除了MAXOUP是2个， 还有MAX_OUP拆成两个
    loop2=PENUM;
    loop3=R_div_2INP; 





	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;
	ap_uint<128> temp_128b;
    unsigned index;


	for (unsigned rep = 0; rep < NumLines; rep++) { 
#pragma HLS PIPELINE II=1
		temp=fifo_out.read();
		temp_128b=temp;

        // 存数按照数据下次用的顺序
        index=loop3_cnt*loop2*loop1*loop0+loop2_cnt*loop1*loop0+loop1_cnt*loop0+loop0_cnt;
		
        ddr_fm_result[rep]=temp_128b;


        // 取数按照数据来的顺序
        if(loop1_cnt==loop1-1){
            loop1_cnt=0;
            if(loop0_cnt==loop0-1){
                loop0_cnt=0;
                if(loop2_cnt==loop2-1){
                    loop2_cnt=0;
                    if(loop3_cnt==loop3-1){
                        loop3_cnt=0;
                    }
                    else{
                        loop3_cnt++;
                    }
                }
                else{
                    loop2_cnt++;
                }
            }
            else{
                loop0_cnt++;
            }
        }
        else{
            loop1_cnt++;
        }
	}
    
}



void Write_to_DDR_NORM_MM_Tranpose(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
    const unsigned PENUM,
    const unsigned R_div_2INP,  // Groups
	const unsigned NumLines){


    unsigned int loop0,loop1,loop2,loop3;

    unsigned int loop0_cnt=0;
    unsigned int loop1_cnt=0;
    unsigned int loop2_cnt=0;
    unsigned int loop3_cnt=0;

    loop0=4;  // 除了MAXOUP是2个， 还有MAX_OUP拆成两个
    loop1=MAX_INP;  
    loop2=R_div_2INP;
    loop3=PENUM;     





	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;
	ap_uint<128> temp_128b;
    unsigned index;


	for (unsigned rep = 0; rep < NumLines; rep++) { 
#pragma HLS PIPELINE II=1
		temp=fifo_out.read();
		temp_128b=temp;

        // 存数按照数据下次用的顺序
        index=loop3_cnt*loop2*loop1*loop0+loop2_cnt*loop1*loop0+loop1_cnt*loop0+loop0_cnt;
		
        ddr_fm_result[rep]=temp_128b;


        // 取数按照数据来的顺序
        if(loop0_cnt==loop0-1){
            loop0_cnt=0;
            if(loop1_cnt==loop1-1){
                loop0_cnt=0;
                if(loop3_cnt==loop3-1){
                    loop3_cnt=0;
                    if(loop2_cnt==loop2-1){
                        loop2_cnt=0;
                    }
                    else{
                        loop2_cnt++;
                    }
                }
                else{
                    loop3_cnt++;
                }
            }
            else{
                loop1_cnt++;
            }
        }
        else{
            loop0_cnt++;
        }
	}
    
}

void Write_Out_to_DDR_NORM_CONV(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
    const unsigned OUT_W,
    const unsigned D,
    const unsigned OUT_H,
	const unsigned M_div_D,
	const unsigned NumLines){


    unsigned int loop0,loop1,loop2,loop3,loop4,loop5;

    unsigned int loop0_cnt=0;
    unsigned int loop1_cnt=0;
    unsigned int loop2_cnt=0;
    unsigned int loop3_cnt=0;
    unsigned int loop4_cnt=0;
    unsigned int loop5_cnt=0;


    loop0=2;  // MAX_OUP本身拆成的2个
    loop1=MAX_INP/MAX_OUP;
    loop2=OUT_W/2;
    loop3=D/MAX_INP; 
    loop4=M_div_D;
    loop5=OUT_H;




	ap_uint<IN_BIT * 2*MAX_OUP/2> temp;
	ap_uint<128> temp_128b;
    unsigned index;


	for (unsigned rep = 0; rep < NumLines; rep++) { 
#pragma HLS PIPELINE II=1
		temp=fifo_out.read();
		temp_128b=temp;

        // 存数按照数据下次用的顺序
        index=loop5_cnt*loop4*loop3*loop2*loop1*loop0+loop4_cnt*loop3*loop2*loop1*loop0+loop3_cnt*loop2*loop1*loop0+loop2_cnt*loop1*loop0+loop1_cnt*loop0+loop0_cnt;
		
        ddr_fm_result[rep]=temp_128b;


        // 取数按照数据来的顺序
        if(loop0_cnt==loop0-1){
            loop0_cnt=0;
            if(loop2_cnt==loop2-1){
                loop2_cnt=0;
                if(loop1_cnt==loop1-1){
                    loop1_cnt=0;
                    if(loop3_cnt==loop3-1){
                        loop3_cnt=0;
                        if(loop5_cnt==loop5-1){
                            loop5_cnt=0;
                            if(loop4_cnt==loop4-1){
                                loop4_cnt=0;
                            }
                            else{
                                loop4_cnt++;
                            }
                        }
                        else{
                            loop5_cnt++;
                        }
                    }
                    else{
                        loop3_cnt++;
                    }
                }
                else{
                    loop1_cnt++;
                }
            }
            else{
                loop2_cnt++;
            }
        }
        else{
            loop0_cnt++;
        }
	}
    
}


void Write_Out_to_DDR_NORM(stream<ap_uint<IN_BIT * 2*MAX_OUP/2> > &fifo_out,
	ap_uint<128>* ddr_fm_result,
    const unsigned PENUM,
    const unsigned OUT_W,
    const unsigned D,
    const unsigned OUT_H,
	const unsigned M_div_D,
	const unsigned NumLines,
    const unsigned which_path,
	const bool skip_mode
	){

    if(skip_mode==false){
      return;
    }




    if(which_path==0||which_path==1||which_path==2){
        Write_Out_to_DDR_NORM_CONV(fifo_out,ddr_fm_result,OUT_W, D,OUT_H,M_div_D, NumLines);
    }
    else if(which_path==4||which_path==5||which_path==9){
        Write_to_DDR_NORM_MM_FM(fifo_out,ddr_fm_result,PENUM, M_div_D, NumLines);
    }
    else if(which_path==7||which_path==10){
        Write_to_DDR_NORM_MM_FM_SOFTMAX_GELU(fifo_out,ddr_fm_result,PENUM, M_div_D, NumLines);
    }
    else if(which_path==6){
        Write_to_DDR_NORM_MM_Tranpose(fifo_out,ddr_fm_result,PENUM, M_div_D, NumLines);
    }
    else{
        Write_Out_to_DDR_DIRECT(fifo_out,ddr_fm_result,NumLines);

    }





    
}







void ExtractPixels_AXI_CONV_DIRECT(
	ap_uint<128>* in,
	stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> >& out_conv,
	const unsigned NumLines,
	const unsigned conv3_group){


		ap_uint<128> act_in;
		ap_uint<384> act_in_384b;

		unsigned int bitIdx=0;
		unsigned int colIdx=0;

		ap_uint< MAX_INP * IN_BIT * PACK_NUM > temp;

		for(unsigned g = 0; g < conv3_group; g++){
			for (unsigned rep = 0; rep < NumLines*3; rep++) {
		#pragma HLS PIPELINE II=1
					act_in_384b=act_in_384b>>128;

                    act_in_384b(384-1, 256) = in[rep];

					if(bitIdx==3-1){
						temp=act_in_384b;
						out_conv.write(temp);
					}

					// cout <<"The Value of Var_a: \t" <<temp<< " \t Binary format: \t" <<temp.to_string(2).c_str()<< '\n';
					// cout<<"rep:"<<rep<<"  value:"<<temp<<endl;
					
					if(bitIdx==3-1){
						bitIdx=0;
					}
					else{
						bitIdx++;
					}

			}	
		}	

}



void ExtractPixels_AXI_CONV_OUT_TO_IN(
	ap_uint<128>* in,
	stream<ap_uint<MAX_INP * IN_BIT * PACK_NUM> >& out_conv,
	const unsigned NumLines,
	const unsigned conv3_group){


		ap_uint<128> act_in_128b;

        ap_uint<IN_BIT * PACK_NUM*MAX_OUP/2> act_in_80b;

        unsigned int loop0;
		unsigned int bitIdx=0;

        loop0=2*(MAX_INP/MAX_OUP);

		ap_uint< MAX_INP * IN_BIT * PACK_NUM > temp;

		for(unsigned g = 0; g < conv3_group; g++){
			for (unsigned rep = 0; rep < NumLines*4; rep++) {
		#pragma HLS PIPELINE II=1

					temp=temp>>(IN_BIT * PACK_NUM*MAX_OUP/2);
                    act_in_128b = in[rep];

					act_in_80b=act_in_128b;
					temp(320-1,240)=act_in_80b;

					if(bitIdx==loop0-1){
						out_conv.write(temp);
					}

					// cout <<"The Value of Var_a: \t" <<temp<< " \t Binary format: \t" <<temp.to_string(2).c_str()<< '\n';
					// cout<<"rep:"<<rep<<"  value:"<<temp<<endl;
					
					if(bitIdx==loop0-1){
						bitIdx=0;
					}
					else{
						bitIdx++;
					}

			}	
		}	

}





void WriteBiasScaleParam(ap_uint<AXI_BIAS_BIT>* ddr_conv3_bias_scale,
    ap_int<BIAS_BIT> conv3_bias[MAX_OUP][MAX_CONV3_BIAS_LENGTH],
    ap_int<BIAS_BIT> scale_factor[MAX_SCALE_FACTOR_LENGTH],
    unsigned NumLines
){

    unsigned int bitIdx=0;
    unsigned int colIdx=0;
    unsigned int depthIdx=0;

ap_uint<AXI_BIAS_BIT> temp_axi_data;

    for(unsigned i=0; i<NumLines;i++){
#pragma HLS PIPELINE II=1
        temp_axi_data=ddr_conv3_bias_scale[i];

        for(int c=0; c<AXI_BIAS_BIT/BIAS_BIT; c++){
            conv3_bias[colIdx][depthIdx] = temp_axi_data.range(BIAS_BIT*(c+1)-1, BIAS_BIT*c);

            if(colIdx==MAX_OUP-1){
                colIdx=0;
                if(depthIdx==MAX_CONV3_BIAS_LENGTH-1){
                    depthIdx=0;
                }
                else{
                    depthIdx++;
                }
            }
            else{
                colIdx++;
            }
        }



    }

    scale_factor[0]=ap_int<BIAS_BIT>(ddr_conv3_bias_scale[NumLines-1]);

#ifdef OUTPUT_DEBUG
    FILE* fp_bias = fopen("bias_fpga.txt", "wb");
    for(int j=0;j<MAX_CONV3_BIAS_LENGTH;j++){
        for(int i=0;i<MAX_OUP;i++){
            ap_int<BIAS_BIT> tmp=conv3_bias[i][j];
            fprintf(fp_bias, "%d\t", (int)tmp);
        }
        fprintf(fp_bias, "\n");
    }
    fclose(fp_bias);	
#endif

}



void WriteConv3WeightParam(ap_uint<128> *conv3_weight,
    ap_uint<MAX_INP * CONV_K *W_BIT> conv3_w_buffer[MAX_A_COL][MAX_CONV3_WEIGHT_LENGTH],
    unsigned NumLines
){
#pragma HLS INLINE OFF
    ap_uint<128> temp_w_128b[2];
    ap_uint<256> temp_w_256b;
    ap_uint<MAX_INP * CONV_K *W_BIT> temp_w;
    unsigned int bitIdx=0;
    unsigned int colIdx=0;
    unsigned int depthIdx=0;

    for(unsigned i=0; i<NumLines*2;i++){
#pragma HLS PIPELINE II=1

        temp_w_128b[bitIdx]=conv3_weight[i];
        
        // cout <<"The Value of Var_w: \t" <<temp_w<< " \t Binary format: \t" <<temp_w.to_string(2).c_str()<< '\n';
        if(bitIdx==2-1){
            temp_w_256b=(temp_w_128b[1],temp_w_128b[0]);
            temp_w=temp_w_256b;
            conv3_w_buffer[colIdx][depthIdx]=temp_w;
        }

            #ifdef OUTPUTW_DEBUG
                // ap_uint<MAX_INP*CONV_K *W_BIT> test_w;
                // test_w=(temp_w[2],temp_w[1],temp_w[0]);
                cout<<temp_w<<endl;

            #endif



        if(bitIdx==2-1){
            bitIdx=0;
            if(colIdx==MAX_A_COL-1){
                colIdx=0;
                if(depthIdx==MAX_CONV3_WEIGHT_LENGTH-1){
                    depthIdx=0;
                }
                else{
                    depthIdx++;
                }
            }
            else{
                colIdx++;
            }
        }
        else{
            bitIdx++;
        }
    }




}



void CONV3WeightParam_Stream(ap_uint< MAX_INP * CONV_K *W_BIT> conv3_weight_buf[MAX_A_COL][MAX_CONV3_WEIGHT_LENGTH],
	stream<ap_uint<MAX_INP * CONV_K *W_BIT> >  fifo_W_in[MAX_A_COL],
	const unsigned NumLines,
	const unsigned OUT_H,
	bool tran_en){
#pragma HLS INLINE OFF
	if (!tran_en) return;

	ap_uint< MAX_INP * CONV_K *W_BIT > temp;

	for(unsigned j=0; j<OUT_H;j++){
		for(unsigned i=0; i<NumLines;i++){
		#pragma HLS PIPELINE II=1
			for(unsigned m=0;m<MAX_A_COL;m++){
				temp = conv3_weight_buf[m][i];
				// cout<<temp<<endl;
				fifo_W_in[m].write(temp);

                // for(int t=0;t<16*3;t++){
                //     ap_int<4> test=temp((t+1)*4-1,t*4);
                //     cout<<"test: "<<test<<endl;
                // }
			}
		}
	}
}

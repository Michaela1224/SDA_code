#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
using namespace std;

//#define STREAM_DEBUG

// #define TEST_DEBUG
// #define DEQUAN_DEBUG

#define MAX(x, y) (((x) > (y)) ? (x) : (y)) /* \brief Maximum value between x and y*/
#define MAX_BUF_LENGTH 1536  // 02-23-setting true
#define MAX_W 32  // 02-23-setting true 实际为W/2




template <unsigned PaddingUp,
      unsigned PaddingDown,
      unsigned SIMD,
      unsigned PACK_NUM,
			unsigned IN_BIT
>
void SAMEPAD_DSPopt_SA_UP_DOWN(
	stream<ap_uint<SIMD * PACK_NUM * IN_BIT> >& in,
	stream<ap_uint<SIMD * PACK_NUM * IN_BIT> >& out,
  const unsigned Din_H,
  const unsigned Din_W_TRUE,
  const unsigned Cin,
  const unsigned conv3_groups,
  const bool skip_mode
  ){

    if(skip_mode==false){
      return;
    }

    ap_uint<SIMD * PACK_NUM * IN_BIT> outData;
    ap_uint<SIMD * PACK_NUM * IN_BIT> inData;

    for(unsigned int g=0;g<conv3_groups;g++){
        for(unsigned int y = 0; y<Din_H; y++){
          for(unsigned int k = 0; k<Cin/SIMD; k++){
            for(unsigned int x=0; x < Din_W_TRUE; x++){
      #pragma HLS PIPELINE II=1  
              // padding rows
              if(y< PaddingUp||y>=Din_H-PaddingDown){
                outData = 0;
              }
              else{
                outData=in.read();
              }
              out.write(outData);
            }
          }
        }
    }
}


template <unsigned K,unsigned IN_BIT, unsigned SIMD> // 注意这里的IN_H是padding后的
void conv3padding_opt_SA(stream<ap_uint<SIMD * IN_BIT * 2> > &in,
                     stream<ap_uint<SIMD * IN_BIT * 2> > &out,
                     const unsigned IN_H,
                     const unsigned IN_W,
                     const unsigned OUT_H,
                     const unsigned IN_CH,
                     const unsigned OUTPENUM,
                     const unsigned GROUPS,
                     const bool skip_mode) {

    if(skip_mode==false){
      return;
    }


  const unsigned int multiplying_factor = IN_CH/SIMD;
  const unsigned int number_blocks = K + 1 ;

  ap_uint<SIMD * IN_BIT * 2> row_buffer[4][MAX_BUF_LENGTH];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p

  const unsigned int cycles_write_block = OUTPENUM * (IN_W/2) * K *multiplying_factor; // 一次读一行的三个
  const unsigned int cycles_read_block = (IN_W/2)*multiplying_factor;
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
  const unsigned int baseIter = (IN_W/2) * K *multiplying_factor // Initial buffer
			                  + OUT_H * MAX(cycles_write_block,cycles_read_block);

  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, wMat =0,count_simd=0;
  unsigned int counter_internal_block = 0;
  ap_uint<2> current_block_write = 0;
  ap_uint<2> current_block_read = 0;

  ap_uint<2> block_read_K;

  unsigned int current_line = 0;
  unsigned int current_line_w = 0;
  unsigned int current_line_simd = 0;
  unsigned int read_block = 0; 
  unsigned int current_line_in_block;
  // unsigned int  flag = 0; 
  ap_uint<SIMD * IN_BIT * 2> inElem;
  ap_uint<2 * SIMD * IN_BIT> data;
  #ifdef INPAD_DEBUG
    unsigned int m=0;
  #endif
  for(unsigned int g=0;g<GROUPS;g++){
    for (unsigned rep = 0; rep < baseIter; rep++) {
  #pragma HLS PIPELINE II=1   
      if (inp < K* (IN_W/2)*multiplying_factor) {// Initial buffer of ConvKernelDim lines	
          inElem = in.read();

          row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;
          inp++;

          if(current_line_w==IN_W/2-1){
              current_line_w=0;
              if(current_line_simd==multiplying_factor-1){
                current_line_simd=0;
                read_block++;
                current_block_write++;
              }
              else{
                current_line_simd++;
              }
          }
          else{
            current_line_w++;
          }
      }
      else{
        if(counter_internal_block < cycles_write_block){
          block_read_K=current_block_read+k_y;


          current_line_in_block = ofm_x*multiplying_factor+count_simd;

          data=row_buffer[block_read_K][(current_line_in_block)];


          out.write(data);
          #ifdef INPAD_DEBUG
              if(m==10768){
                cout<<"debug...."<<endl;
                cout<<data<<endl;
              }
              m++;
          #endif

          if(ofm_x==IN_W/2-1){
            ofm_x=0;
            if(count_simd==multiplying_factor-1){
              count_simd=0;
              if(k_y==K-1){
                k_y=0;
                if(wMat==OUTPENUM-1){
                  wMat=0;
                  current_block_read++;
                }
                else{
                  wMat++;
                }              
              }
              else{
                k_y++;
              }
            }
            else{
              count_simd++;
            }
          }
          else{
            ofm_x++;
          }    
        }
        if ((counter_internal_block < cycles_read_block) && (read_block<IN_H)) {
          inElem=in.read();
          row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;


          if(current_line_w==IN_W/2-1){
              current_line_w=0;
              if(current_line_simd==multiplying_factor-1){
                current_line_simd=0;
                read_block++;
                current_block_write++;

              }
              else{
                current_line_simd++;
              }
          }
          else{
            current_line_w++;
          }
        }


        if(counter_internal_block == (max_cycles-1)){
          counter_internal_block = 0;
        }
        else{
          counter_internal_block++; 
        }

      }
      if(rep==baseIter-1){
            inp=0;
            read_block=0;
            current_block_write=0;
            current_block_read=0;
      }
    }
  }
}



template <unsigned A_Row, unsigned A_Col,	
      unsigned PE, unsigned SA_PE, unsigned PACK_OUT_NUM, unsigned ACC_BIT>
void MM_to_Out(
	stream<ap_uint<ACC_BIT*PACK_OUT_NUM> > in[A_Row][A_Col][SA_PE],
	stream<ap_uint<ACC_BIT*PACK_OUT_NUM*A_Row> > out[PE],
  const unsigned NumLines,
  const unsigned skip_mode){


    if(skip_mode==1){
      return;
    }
    
    ap_uint<ACC_BIT*PACK_OUT_NUM*A_Row*A_Col> result;
    ap_uint<ACC_BIT*PACK_OUT_NUM> tmp;

    for (unsigned long long rep = 0; rep < NumLines; rep++) {
#pragma HLS PIPELINE II=1  
      for(unsigned int c = 0; c < A_Col; c++){ 
        for(unsigned int y = 0; y < SA_PE; y++){
              for(unsigned int r = 0; r < A_Row; r++){
                result=result>>ACC_BIT*PACK_OUT_NUM;
                tmp=in[r][c][y].read();
                result(ACC_BIT*PACK_OUT_NUM*A_Row*A_Col-1,ACC_BIT*PACK_OUT_NUM*A_Row*A_Col-ACC_BIT*PACK_OUT_NUM)=tmp;
              }
              out[c*SA_PE+y].write(result);
            }
          }
    }
    
  }


template <unsigned A_Row, unsigned A_Col,	
      unsigned PE, unsigned SA_PE, unsigned PACK_OUT_NUM, unsigned ACC_BIT>
void MM_Parallel_to_Serial_Out(
	stream<ap_uint<ACC_BIT*PACK_OUT_NUM> > in[A_Row][A_Col][SA_PE],
	stream<ap_uint<ACC_BIT*PACK_OUT_NUM> > out[PE],
  const unsigned NumLines,
  const bool skip_mode){


    if(skip_mode==true){
      return;
    }
    
    // ap_uint<ACC_BIT*PACK_OUT_NUM*A_Row*A_Col> result;
    ap_uint<ACC_BIT*PACK_OUT_NUM> tmp;

    for (unsigned long long rep = 0; rep < NumLines; rep++) {
      for(unsigned int r = 0; r < A_Row; r++){
#pragma HLS PIPELINE II=1  
        for(unsigned int c = 0; c < A_Col; c++){ 
          for(unsigned int y = 0; y < SA_PE; y++){
                // result=result>>ACC_BIT*PACK_OUT_NUM;
                tmp=in[r][c][y].read();
                out[c*SA_PE+y].write(tmp);
                // result(ACC_BIT*PACK_OUT_NUM*A_Row*A_Col-1,ACC_BIT*PACK_OUT_NUM*A_Row*A_Col-ACC_BIT*PACK_OUT_NUM)=tmp;
              }
              
            }
          }
    }
    
  }



template <unsigned A_Row, unsigned A_Col,	unsigned PE, unsigned SA_PE,unsigned SIMD, unsigned SA_SIMD,
          unsigned CONV_K, unsigned MAX_LENGTH, unsigned W_BIT>
void W_conv3_array(
              stream<ap_uint<SIMD * CONV_K *W_BIT> >  fifo_W_in[A_Col],
                stream<ap_uint<SA_SIMD * CONV_K * W_BIT> > fifo_W_local_out[A_Row][A_Col],
                const unsigned OUT_H,
                const unsigned NumLines,
                const unsigned GROUPS,
						    const bool skip_mode) {
#pragma HLS INLINE OFF


    if(skip_mode==false){
      return;
    }

    ap_uint<SIMD * CONV_K *W_BIT>  w;
    ap_uint<SA_SIMD * CONV_K *W_BIT>  temp;

#ifdef DEBUG
  FILE* fp_win0 = fopen("W3_reorg_SIMD_all_in.txt", "wb");
#endif


  for (unsigned int h = 0; h < GROUPS; h++) { // 40
    for (unsigned int peIdx = 0; peIdx < OUT_H*NumLines; peIdx++) {
#pragma HLS PIPELINE II=1
          for (unsigned int c = 0; c < A_Col; c++) {
              w=fifo_W_in[c].read();
              for(unsigned int r = 0;  r< A_Row; r++) {
                temp=w((r+1)*SA_SIMD * CONV_K *W_BIT-1,r*SA_SIMD*CONV_K *W_BIT);
                fifo_W_local_out[r][c].write(temp);
              }
          }
        }
    }


#ifdef DEBUG
  fclose(fp_win0);
#endif

}



template <unsigned A_Row, unsigned A_Col,	 unsigned InStreamW,
      unsigned OutStreamW>
void A_to_array(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > out[A_Row][A_Col],
  const unsigned NumLines){
  ap_uint<OutStreamW> temp_row;


#ifdef STREAM_DEBUG
    FILE* fp_asimd= fopen("conv3_stream_A.txt", "wb");
#endif

	for (unsigned long long rep = 0; rep < NumLines; rep++) {
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=1
		ap_uint<InStreamW> temp = in.read();

    #ifdef STREAM_DEBUG
        for(int i=0;i<8;i++){
          ap_uint<4> test=temp((i+1)*4-1,i*4);
          fprintf(fp_asimd, "%d\n", (int)test);
        }
    #endif

    for(unsigned int r = 0; r < A_Row; r++){
      temp_row=temp((r+1)*OutStreamW-1,r*OutStreamW);
      for(unsigned int c = 0; c < A_Col; c++){
        out[r][c].write(temp_row);
      }
    }
	}


#ifdef STREAM_DEBUG
   fclose(fp_asimd);
#endif

}



template <unsigned A_Row, unsigned A_Col,	 unsigned InStreamW,
      unsigned OutStreamW>
void W_mm_to_array(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > out[A_Row][A_Col],
  const unsigned NumLines,
  const bool skip_mode){

    if(skip_mode==true){
      return;
    }


  ap_uint<OutStreamW> temp_row;


#ifdef STREAM_DEBUG
    FILE* fp_asimd= fopen("conv3_stream_A.txt", "wb");
#endif

	for (unsigned long long rep = 0; rep < NumLines; rep++) {
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=1
		ap_uint<InStreamW> temp = in.read();

    #ifdef STREAM_DEBUG
        for(int i=0;i<8;i++){
          ap_uint<4> test=temp((i+1)*4-1,i*4);
          fprintf(fp_asimd, "%d\n", (int)test);
        }
    #endif
    for(unsigned int c = 0; c < A_Col; c++){
      temp_row=temp((c+1)*OutStreamW-1,c*OutStreamW);
        for(unsigned int r = 0; r < A_Row; r++){
        out[r][c].write(temp_row);
      }
    }
	}
#ifdef STREAM_DEBUG
   fclose(fp_asimd);
#endif

}



ap_uint<44> correct_fun(ap_uint<4> w0,ap_uint<4> w1,ap_uint<4> w2, ap_uint<4> a0,ap_uint<4> a1){
#pragma HLS INLINE

	ap_uint<44> C_port;
	ap_uint<1> signw0=w0.range(3,3);
	ap_uint<1> signw1=w1.range(3,3);
	ap_uint<1> signw2=w2.range(3,3);
	ap_uint<1> signa0=a0.range(3,3);
	ap_uint<1> signa1=a1.range(3,3);

	ap_uint<4> sign4w0=(signw0,signw0,signw0,signw0);
	ap_uint<4> sign4w1=(signw1,signw1,signw1,signw1);
	ap_uint<4> sign4w2=(signw2,signw2,signw2,signw2);
	ap_uint<4> sign4a0=(signa0,signa0,signa0,signa0);
	ap_uint<4> sign4a1=(signa1,signa1,signa1,signa1);

	ap_uint<4> out0_L0=(sign4w2&a0)+(sign4a0&w2);
	ap_uint<8>  res0_correct=ap_uint<8>(out0_L0)<<4;

	#ifdef DEBUG 
		std::cout <<"The Value of res0_correct: \t" <<res0_correct << "\t Binary format: \t" <<res0_correct.to_string(2).c_str()<< '\n';
	#endif

	ap_uint<4> out0_L10=(sign4w1&a0)+(sign4a0&w1);

    ap_uint<4> out0_L11=(sign4w2&a1)+(sign4a1&w2);

	ap_uint<9>  res1_correct=ap_uint<9> (out0_L10+out0_L11)<<4;

	#ifdef DEBUG 
		std::cout <<"The Value of res1_correct: \t" <<res1_correct << "\t Binary format: \t" <<res1_correct.to_string(2).c_str()<< '\n';
	#endif

	ap_uint<4> out0_L20=(sign4w0&a0)+(sign4a0&w0);
    ap_uint<4> out0_L21=(sign4w1&a1)+(sign4a1&w1);


	ap_uint<9>  res2_correct=ap_uint<9> (out0_L20+out0_L21)<<4; 

	#ifdef DEBUG 
		std::cout <<"The Value of res2_correct: \t" <<res2_correct << "\t Binary format: \t" <<res2_correct.to_string(2).c_str()<< '\n';
	#endif


	ap_uint<4> out0_L3=(sign4w0&a1)+(sign4a1&w0);
	ap_uint<8>  res3_correct=ap_uint<8> (out0_L3)<<4; 

	#ifdef DEBUG 
		std::cout <<"The Value of res3_correct: \t" <<res3_correct << "\t Binary format: \t" <<res3_correct.to_string(2).c_str()<< '\n';
	#endif

	C_port= ((ap_uint<44>)res3_correct<<33)+((ap_uint<33> )res2_correct<<22)+((ap_uint<22>)res1_correct<<11)+res0_correct;

	return C_port;

}


void RMPacking_4b_SignedA(ap_uint<4> A0,ap_uint<4> A1,ap_uint<4> W0,ap_uint<4> W1,ap_uint<4> W2, ap_int<8> result[4]){

  ap_uint<15> B_port= ((ap_uint<15>)A1<<11)+A0;
  ap_uint<26> A_port=((ap_uint<26>)W0<<22)+((ap_uint<22>)W1<<11);
  ap_uint<4> D_port=W2;

  // result correction
  ap_uint<44> C_port=correct_fun(W0,W1,W2,A0,A1);  

  // DSP computation
  ap_uint<44> P_port=(A_port+D_port)*B_port-C_port;

  ap_uint<11> out[4];

  // 44-bit data split
  out[0]=P_port(11-1,0);  // w2a0 for conv | w1a0 for MM
  out[1]=P_port(22-1,11-1);  // w1a0+w2a1 for conv | w1a1 for MM
  out[2]=P_port(33-1,22-1);  // w0a0+w1*a1 for conv | w0a0 for MM
  out[3]=P_port(44-1,33-1);   // w0a1 for conv | w0a1 for MM

  result[0]=ap_int<8>(out[0]);
  for(int x=1;x<4;x++){
    out[x]=(out[x]>>1)+(out[x]&1);
    result[x]=ap_int<8>(out[x]);
  }


  #ifdef INPUT_DEBUG 
    ap_int<4> test_A0,test_A1, test_W0, test_W1,test_W2;
    test_A0=(ap_int<4>)A0;
    test_A1=(ap_int<4>)A1;
    test_W0=(ap_int<4>)W0;
    test_W1=(ap_int<4>)W1;
    test_W2=(ap_int<4>)W2;
    cout <<"The Value of A0: \t" <<test_A0 << endl;
    cout <<"The Value of A1: \t" <<test_A1 << endl;
    cout <<"The Value of W0: \t" <<test_W0<< endl;
    cout <<"The Value of W1: \t" <<test_W1<< endl;
    cout <<"The Value of W2: \t" <<test_W2<< endl;
    ap_int<9> test_out0,test_out1,test_out2,test_out3;
    test_out0=test_W2*test_A0;
    test_out1=test_W1*test_A0+test_W2*test_A1;
    test_out2=test_W0*test_A0+test_W1*test_A1;
    test_out3=test_W0*test_A1;
    cout <<"test_out0: \t" <<test_out0<<"\t real_out0: \t" <<ap_int<8>(result[0])<< endl;
    cout <<"test_out1: \t" <<test_out1<<"\t real_out1: \t" <<ap_int<8>(result[1])<< endl;
    cout <<"test_out2: \t" <<test_out2<<"\t real_out2: \t" <<ap_int<8>(result[2])<< endl;
    cout <<"test_out3: \t" <<test_out3<<"\t real_out3: \t" <<ap_int<8>(result[3])<< endl;

    if((ap_int<8>(out[0])!=test_out0)||(ap_int<8>(out[1])!=test_out1)||(ap_int<8>(out[2])!=test_out2)||(ap_int<8>(out[3])!=test_out3)){
      cout<<"debug error";
    }

  #endif    



}



void RMPacking_4b_USignedA(ap_uint<4> A0,ap_uint<4> A1,ap_int<4> W0,ap_int<4> W1,ap_int<4> W2, ap_int<8> result[4]){



  ap_uint<15> B_port=(A1(4-1,0), (ap_uint<11 - 4>)0,A0(4-1,0));
  ap_int<26> D_port=W0*(1<<(22))+W1*(1<<11)+W2;


  // DSP computation
  ap_int<44> P_port=D_port*B_port;

  ap_int<11> out0;
  ap_int<12> out1;
  ap_int<12> out2;
  ap_int<12> out3;

  // 44-bit data split
  out0=P_port(11-1,0);  // w2a0 for conv | w1a0 for MM
  out1=P_port(22-1,11-1);  // w1a0+w2a1 for conv | w1a1 for MM
  out2=P_port(33-1,22-1);  // w0a0+w1*a1 for conv | w0a0 for MM
  out3=P_port(44-1,33-1);   // w0a1 for conv | w0a1 for MM

  result[0]=ap_int<11>(out0);
  result[1]=(out1 >> 1) + (out1 & 1);
  result[2]=(out2 >> 1) + (out2 & 1);
  result[3]=(out3 >> 1) + (out3 & 1);


  #ifdef INPUT_DEBUG 
    ap_uint<4> test_A0,test_A1; 
    ap_int<9> test_W0, test_W1,test_W2;
    test_A0=(ap_uint<4>)A0;
    test_A1=(ap_uint<4>)A1;
    test_W0=(ap_int<4>)W0;
    test_W1=(ap_int<4>)W1;
    test_W2=(ap_int<4>)W2;
    cout <<"The Value of A0: \t" <<test_A0 << endl;
    cout <<"The Value of A1: \t" <<test_A1 << endl;
    cout <<"The Value of W0: \t" <<test_W0<< endl;
    cout <<"The Value of W1: \t" <<test_W1<< endl;
    cout <<"The Value of W2: \t" <<test_W2<< endl;
    ap_int<9> test_out0,test_out1,test_out2,test_out3;
    test_out0=test_W2*test_A0;
    test_out1=test_W1*test_A0+test_W2*test_A1;
    test_out2=test_W0*test_A0+test_W1*test_A1;
    test_out3=test_W0*test_A1;
    cout <<"test_out0: \t" <<test_out0<<"\t real_out0: \t" <<result[0]<< endl;
    cout <<"test_out1: \t" <<test_out1<<"\t real_out1: \t" <<result[1]<< endl;
    cout <<"test_out2: \t" <<test_out2<<"\t real_out2: \t" <<result[2]<< endl;
    cout <<"test_out3: \t" <<test_out3<<"\t real_out3: \t" <<result[3]<< endl;

    if((result[0]!=test_out0)||(result[1]!=test_out1)||(result[2]!=test_out2)||(result[3]!=test_out3)){
      cout<<"debug error";
    }

  #endif    



}






template <unsigned IN_BIT, unsigned W_BIT, unsigned PACK_NUM,unsigned PACK_CONV_NUM,unsigned PACK_OUT_NUM,
          unsigned SIMD,  unsigned PE, unsigned ACC_BIT>
void PE_wrapper(int idr, int idc, stream<ap_uint<SIMD*PACK_NUM*IN_BIT> > &fifo_A_in,
                stream<ap_uint<SIMD*PACK_CONV_NUM*W_BIT> > &fifo_W_in, 
                stream<ap_uint<ACC_BIT*PACK_OUT_NUM> > fifo_C_out[PE],
                const unsigned NWnum,
                const unsigned NumLines,
                const bool mode
                ) {
// mode=0 OS stationary for matrix
// mode=1 WS stationary for conv3
 
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=fifo_C_out dim=1 complete

    ap_uint<SIMD * PACK_NUM *IN_BIT> A_simd_reg[SIMD];
#pragma HLS ARRAY_PARTITION variable=A_simd_reg dim=1 complete

    ap_uint<PACK_NUM*IN_BIT> data_A_reg[SIMD][PE];
#pragma HLS ARRAY_PARTITION variable=data_A_reg dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data_A_reg dim=2 complete


    ap_uint<PE * PACK_CONV_NUM *W_BIT> W_pe_reg[PE];
#pragma HLS ARRAY_PARTITION variable=W_pe_reg dim=1 complete

    ap_uint<PACK_CONV_NUM*W_BIT> data_W_reg[SIMD][PE];
#pragma HLS ARRAY_PARTITION variable=data_W_reg dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data_W_reg dim=2 complete


//     ap_int<ACC_BIT> OS_ACC_reg[SIMD][PE][PACK_OUT_NUM];
// #pragma HLS ARRAY_PARTITION variable=OS_ACC_reg dim=1 complete
// #pragma HLS ARRAY_PARTITION variable=OS_ACC_reg dim=2 complete
// #pragma HLS ARRAY_PARTITION variable=OS_ACC_reg dim=3 complete

    ap_uint<ACC_BIT*PACK_OUT_NUM> res_C_reg[PE];
#pragma HLS ARRAY_PARTITION variable=res_C_reg dim=1 complete


    ap_int<ACC_BIT> data_C_reg[SIMD][PE][PACK_OUT_NUM];
#pragma HLS ARRAY_PARTITION variable=data_C_reg dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data_C_reg dim=2 complete
#pragma HLS ARRAY_PARTITION variable=data_C_reg dim=3 complete

#ifdef STREAM_DEBUG
    // FILE* fp_pe00_a = fopen("a_stream_pe00.txt", "wb");
    // FILE* fp_pe00_w= fopen("w_stream_pe00.txt", "wb");
    // FILE* fp_pe01_a = fopen("a_stream_pe11.txt", "wb");
    // FILE* fp_pe01_w= fopen("w_stream_pe11.txt", "wb");
    FILE* fp_pe00_res= fopen("res_stream_pex3.txt", "wb");

#endif

    ap_int<ACC_BIT> acc_tmp[SIMD][PE][4];
#pragma HLS ARRAY_PARTITION variable=acc_tmp dim=1 complete
#pragma HLS ARRAY_PARTITION variable=acc_tmp dim=2 complete
#pragma HLS ARRAY_PARTITION variable=acc_tmp dim=3 complete

  int rn_index=0;
  int out_flag=0;
//  int flag=0;
  // int cascade_index=0;

  for(unsigned i=0; i<SIMD; i++){
#pragma HLS UNROLL
    A_simd_reg[i]=0;
  }

  for(unsigned i=0; i<PE; i++){
#pragma HLS UNROLL
    W_pe_reg[i]=0;
  }

  for(unsigned j=0; j<PE;j++){
#pragma HLS UNROLL
    for(unsigned i=0; i<SIMD; i++){
#pragma HLS UNROLL
      data_A_reg[i][j]=0;
      data_W_reg[i][j]=0;
    }
  }

  for(unsigned j=0; j<PE;j++){
#pragma HLS UNROLL
    for(unsigned i=0; i<SIMD; i++){
#pragma HLS UNROLL
      for(unsigned m=0; m<PACK_OUT_NUM; m++){
#pragma HLS UNROLL
        data_C_reg[i][j][m]=0;
      }
    }
  }


  for (unsigned rep = 0; rep < NumLines+PE+SIMD-2; rep++) { // 40
#pragma HLS PIPELINE II=1


            #ifdef INPUT_DEBUG 
              if(rep==14){
                cout<<"start check....................."<<endl;
              }
            #endif  

    if(rep<NumLines){
      A_simd_reg[0]=fifo_A_in.read();  // 激活取数后面对不对
      // W_pe_reg[0]=fifo_W_in.read();
    }
    else{
      A_simd_reg[0]=0;
      // W_pe_reg[0]=0;
    }

    if(rep<NumLines){
        if(mode==false){
            W_pe_reg[0]=fifo_W_in.read();
        }
        else if((mode==true) &&(rn_index<PE)){
            W_pe_reg[0]=fifo_W_in.read();
        } 
    }
    else if (rep>=NumLines){
        W_pe_reg[0]=0;
    }  
  

// A fetcher WS/ OS mode share
    for(unsigned m=0; m<SIMD;m++){
    #pragma HLS UNROLL
      ap_uint<IN_BIT*PACK_NUM > temp_a;
      temp_a=A_simd_reg[m](PACK_NUM*IN_BIT-1,0);
      data_A_reg[m][0]=temp_a;
      A_simd_reg[m]=A_simd_reg[m]>>(PACK_NUM*IN_BIT);
    }

// W fetcher
// OS mode:  write W to the top row PE00-PE03 every cycle
// WS mode: in each first PE+SIMD-1 cycle, write W to PE00-PE33
    for(unsigned k=0; k<PE;k++){   // 共用一次右移对不对
    #pragma HLS UNROLL
        ap_uint<W_BIT*PACK_CONV_NUM> temp_w;
        temp_w=W_pe_reg[k](PACK_CONV_NUM*W_BIT-1,0);
        if(mode==false){
            data_W_reg[0][k]=temp_w;
        }
        else if((mode==true) &&(rn_index<PE+SIMD-1)){
            if((rn_index-k>=0) && (rn_index-k<=SIMD-1)){
                data_W_reg[rn_index-k][k]=temp_w;
            }
            else if( (NWnum<PE+SIMD-1)&&(rn_index+PE-k>=0) && (rn_index+PE-k<=SIMD-1) ){
              data_W_reg[rn_index-k+PE][k]=temp_w;
            }
        }
    // cout <<data_W_reg[0][k]<<endl;
        W_pe_reg[k]=W_pe_reg[k]>>(3*W_BIT);
    }


  // A fetcher /W fetcher  down sliding
      for(unsigned m=SIMD-1; m>0;m--){
      #pragma HLS UNROLL
        A_simd_reg[m]=A_simd_reg[m-1];
      }

      for(unsigned m=PE-1; m>0;m--){
      #pragma HLS UNROLL
        W_pe_reg[m]=W_pe_reg[m-1];
      }


      for (int j=PE-1; j>=0;j--) { // PE
#pragma HLS UNROLL
        for (int i=SIMD-1; i>=0;i--){ // SIMD
#pragma HLS UNROLL

            // Read A，W
            ap_uint<2*IN_BIT> data_A_tmp;
            data_A_tmp= data_A_reg[i][j];

            ap_uint<3*W_BIT> data_W_tmp;
            data_W_tmp= data_W_reg[i][j];



            for(int x=0;x<4;x++){
              if(mode==false){ // OS mode: read acc result
                if(rep>=NWnum&&i+j==rn_index){
                  acc_tmp[i][j][x]= 0;
                }
                else{
                  acc_tmp[i][j][x]= data_C_reg[i][j][x];
                }

              }
              else if(mode==true){ // WS mode: read top (i-1,j) acc result
                if(i==0){
                    acc_tmp[i][j][x]= 0;
                }
                else{
                    acc_tmp[i][j][x]= data_C_reg[i-1][j][x];
                }
              }
            }


            ap_uint<IN_BIT> A0,A1;
            ap_uint<W_BIT> W0,W1,W2;
            (A1, A0) = data_A_tmp;
            ap_uint<4> A00,A01,A10,A11;

            (A01, A00) = A0;  (A11, A10) = A1;  
            (W2, W1, W0) = data_W_tmp;   

            #ifdef TEST_DEBUG 

              std::cout <<"The Value of A0: \t" <<(ap_int<IN_BIT>)A0 << "\t Binary format: \t" <<A0.to_string(2).c_str()<< '\n';
              std::cout <<"The Value of A1: \t" <<(ap_int<IN_BIT>)A1 << "\t Binary format: \t" <<A1.to_string(2).c_str()<< '\n';
              std::cout <<"The Value of W0: \t" <<(ap_int<W_BIT>)W0 << "\t Binary format: \t" <<W0.to_string(2).c_str()<< '\n';
              std::cout <<"The Value of W1: \t" <<(ap_int<W_BIT>)W1 << "\t Binary format: \t" <<W1.to_string(2).c_str()<< '\n';
              std::cout <<"The Value of W2: \t" <<(ap_int<W_BIT>)W2 << "\t Binary format: \t" <<W2.to_string(2).c_str()<< '\n';
            #endif  

            ap_int<8> out_L[4], out_H[4];
            ap_int<13> out_T[4];

            RMPacking_4b_USignedA(A00,A10, W0,W1,W2, out_L);
            RMPacking_4b_SignedA(A01,A11, W0,W1,W2, out_H);


            // 数据累加
            for(int x=0;x<4;x++){
            #pragma HLS UNROLL
  
              // cout<<"out_L:"<<(ap_int<8>)out_L[x]<<endl;
              // cout<<"out_H:"<<(ap_int<8>)out_H[x]<<endl;

              out_T[x]=((ap_int<8>)out_L[x])+(((ap_int<13>)out_H[x])<<4);


              // cout<<"out_T:"<<(ap_int<13>)out_T[x]<<endl;


              data_C_reg[i][j][x]= acc_tmp[i][j][x]+out_T[x];
            }              


            //WS / OS mode: A right-pass
            if(j<PE-1){
              data_A_reg[i][j+1]=data_A_tmp;
            }
            //OS: W down-pass
            if(mode==false&&i<SIMD-1){
              data_W_reg[i+1][j]=data_W_tmp;
            }

            // OS mode: reg final result
            if(i+j==out_flag){
              ap_uint<ACC_BIT*4> out_tmp=(data_C_reg[i][j][1],data_C_reg[i][j][0],data_C_reg[i][j][3],data_C_reg[i][j][2]);
              res_C_reg[j]=out_tmp;
            }


          }
        }
        // OS mode: output final result
        if(mode==false){
            if(rep>=NWnum-1&&out_flag<PE+SIMD-1){
                for(int d_j=0;d_j<PE;d_j++){
                    if((d_j<=out_flag&&out_flag<PE)||(d_j>=(out_flag-PE+1)&&out_flag>=PE)){
                        fifo_C_out[d_j].write(res_C_reg[d_j]);
                    }
                }                
            }
        }
         // WS mode: output final result
        else if(mode==true){
            for(int d_j=0;d_j<PE;d_j++){
            #pragma HLS UNROLL
                if ( ((SIMD-1+d_j<=rep) && (rep<NumLines) ) || ( (SIMD-1+d_j>(rep-NumLines)) && (rep>=NumLines)) ){   // 这个条件再重新写一下

                    // cout<<"data_C_reg[SIMD-1][d_j][3]:"<<data_C_reg[SIMD-1][d_j][3]<<endl;
                    // cout<<"data_C_reg[SIMD-1][d_j][2]:"<<data_C_reg[SIMD-1][d_j][2]<<endl;
                    // cout<<"data_C_reg[SIMD-1][d_j][1]:"<<data_C_reg[SIMD-1][d_j][1]<<endl;
                    // cout<<"data_C_reg[SIMD-1][d_j][0]:"<<data_C_reg[SIMD-1][d_j][0]<<endl;
                    // if((data_C_reg[SIMD-1][d_j][3]!=0)||(data_C_reg[SIMD-1][d_j][2]!=0)||(data_C_reg[SIMD-1][d_j][1]!=0)||(data_C_reg[SIMD-1][d_j][0]!=0)){
                    //   cout<<"debug"<<endl;
                    // }

                    ap_uint<ACC_BIT*4> out_tmp=(data_C_reg[SIMD-1][d_j][3],data_C_reg[SIMD-1][d_j][2],data_C_reg[SIMD-1][d_j][1],data_C_reg[SIMD-1][d_j][0]);
                    fifo_C_out[d_j].write(out_tmp); 
                }
            }            
        }



    if(rn_index==NWnum-1){
        rn_index=0;
      }
    else{
      rn_index++;
    }

    if(rn_index==NWnum-1){
        out_flag=0;
      }
    else if(out_flag==PE+SIMD-1){
      out_flag=out_flag;
    }
    else{
      out_flag++;
    }


  }

#ifdef STREAM_DEBUG
      // fclose(fp_pe01_a);
      // fclose(fp_pe01_w);
      fclose(fp_pe00_res);
#endif

}





template <unsigned MAX_A_ROW,unsigned MAX_A_COL,  unsigned PE,unsigned SA_PE, unsigned PACK_OUT_NUM, unsigned M_BIT>
void arrar_acc_to_Res( stream<ap_uint<M_BIT*PACK_OUT_NUM> > fifo_C_in[MAX_A_ROW][MAX_A_COL][SA_PE], stream<ap_uint<M_BIT*PACK_OUT_NUM> > fifo_C_out[PE],
                const unsigned numlines,
                const bool skip_mode){


    if(skip_mode==false){
      return;
    }

ap_uint<M_BIT*PACK_OUT_NUM> temp_4m;
ap_uint<M_BIT*PACK_OUT_NUM> psum_4m;

ap_int<M_BIT> temp;
ap_int<M_BIT> res;

  for (unsigned int h = 0; h < numlines; h++) { // 40
//#pragma HLS loop_tripcount min=OUT_H max=OUT_H
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_A_COL; c++){
        for(unsigned int m = 0; m < SA_PE; m++){
          for(unsigned int r = 0; r < MAX_A_ROW; r++){
            if(r==0){
              psum_4m=fifo_C_in[r][c][m].read();
            }
            else{
              temp_4m=fifo_C_in[r][c][m].read();
              for(unsigned int x=0; x<PACK_OUT_NUM;x++){      
                temp=ap_int<M_BIT>(temp_4m((x+1)*M_BIT-1,x*M_BIT));
                res=ap_int<M_BIT>(psum_4m((x+1)*M_BIT-1,x*M_BIT))+temp;
                psum_4m((x+1)*M_BIT-1,x*M_BIT)=res;
              }                                
            }
          }
          //
          // ap_int<M_BIT> test0,test1,test2,test3;

          // (test0,test1,test2,test3)=psum_4m;
          // cout<<"test0:" <<test0<<endl;
          // cout<<"test1:" <<test1<<endl;
          // cout<<"test2:" <<test2<<endl;
          // cout<<"test3:" <<test3<<endl;
          // if(test0!=0||test1!=0||test2!=0||test3!=0){
          //   cout<<"debug"<<endl;
          // }


          fifo_C_out[c*SA_PE+m].write(psum_4m);
        }
    }
  }


}



template <unsigned K, unsigned PE, unsigned M_BIT>
void PE_DSP_ACC(stream<ap_uint<M_BIT*4> > fifo_C_in[PE], 
                stream<ap_uint<M_BIT*2> > fifo_C_res[PE],
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM,
                const unsigned GROUPS,
                const unsigned skip_mode) {
#pragma HLS INLINE OFF

    if(skip_mode==0){
      return;
    }

ap_int<M_BIT> ACC_P2_prev[PE];
#pragma HLS ARRAY_PARTITION variable=ACC_P2_prev dim=1 complete

ap_int<M_BIT> ACC_P3_prev[PE];
#pragma HLS ARRAY_PARTITION variable=ACC_P3_prev dim=1 complete

ap_int<M_BIT> out0=0;
ap_int<M_BIT> out1=0;
unsigned int Iter_NUM=K*SIMDNUM;

#ifdef IN_DEBUG
      FILE* fpw = fopen("w_40_test.txt", "wb");
      FILE* fpa = fopen("a_40_test.txt", "wb");

#endif
  
    for (unsigned int h = 0; h < OUT_H*GROUPS; h++) { // 40
  #pragma HLS loop_tripcount min=OUT_H max=OUT_H
      for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
        for (unsigned int iter = 0; iter < Iter_NUM; iter++) {  
          for (unsigned int w = 0; w < OUT_W /2; w++) {  // OUT_W / 2   80/2
  #pragma HLS PIPELINE II=1
              for(unsigned int m=0; m < PE; m++ ){
                bool m_clear = (w == 0);
              // read FM-A
                ap_int<M_BIT*4> fifo_data_C;
                fifo_data_C= fifo_C_in[m].read();

                ap_int<M_BIT> S0, S1,S2,S3;
                (S3,S2,S1,S0)=fifo_data_C;

                if (m_clear){ // 1 1
                  out0=ACC_P2_prev[m];
                  out1=S1;
                }
                else{// 0 1
                  out0=S0+ACC_P2_prev[m];
                  out1=S1+ACC_P3_prev[m];
                }
                ACC_P2_prev[m]=S2;
                ACC_P3_prev[m]=S3;


            // cout<<"out0:" <<out0<<endl;
            // cout<<"out1:" <<out1<<endl;

            // if(out0!=0||out1!=0){
            //   cout<<"debug"<<endl;
            // }

                // read 乘累加结果
                fifo_C_res[m].write((out1,out0));  // 上一层的结果
            }

          }
        }
      }
    }

for(unsigned int m=0; m < PE; m++ ){
  out0=ACC_P2_prev[m];
  fifo_C_res[m].write((0, out0));
}

#ifdef IN_DEBUG
    if(idx==0&& idy==0 && idr==1&& idc==0){
      fclose(fpw);
      fclose(fpa);
    }
#endif

#ifdef DEBUG
if(idx==1&& idy==7){
  fclose(fp1);
}
#endif

}


template <unsigned M_BIT, unsigned DeQuan_BIT,unsigned DEQUAN_INTEGER_BIT,unsigned BIAS_BIT, unsigned Shift_Factor>
ap_uint<DeQuan_BIT> DeQuan_Bias_Unit(

  ap_int<M_BIT> acc_in,
  ap_int<BIAS_BIT> Bias,
  ap_int<BIAS_BIT> Layer_Scale){

 #pragma HLS inline off
  ap_int<25> qy= acc_in+Bias;
  
  #ifdef DEQUAN_DEBUG
    cout<<"acc_in: "<<acc_in<<endl;
    cout<<"Bias: "<<Bias<<endl;
    cout<<"Layer_Scale: "<<Layer_Scale<<endl;
    cout<<"Shift_Factor: "<<Shift_Factor<<endl;
  #endif
  ap_fixed<DeQuan_BIT, DEQUAN_INTEGER_BIT> fixp_out;
    
  fixp_out=(ap_fixed<48, 40>(qy*Layer_Scale))>>Shift_Factor;

  #ifdef DEQUAN_DEBUG
    cout<<"fixp_out: "<<fixp_out<<endl;
  #endif
  ap_uint<DeQuan_BIT> out;
  out(DeQuan_BIT-1,0)=fixp_out(DeQuan_BIT-1,0);

  #ifdef DEQUAN_DEBUG
    cout<<"out: "<<out<<endl;
  #endif

  return out;
}


/**
 * 2023-12-20: 等后期需要加量化算子
*/
template <unsigned K, unsigned M_BIT,unsigned MAX_OUP>
void Inter_Reorg_acc_to_Res( stream<ap_uint<M_BIT*2> > fifo_C_in[MAX_OUP], 
                stream<ap_uint<M_BIT*2> > fifo_C_out[MAX_OUP],
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM,
                const unsigned GROUPS,
                const bool skip_mode) {


    if(skip_mode==false){
      return;
    }

unsigned int total_num=(GROUPS)*OUT_H*PENUM*K*SIMDNUM*(OUT_W/2);

  ap_uint<M_BIT> data0, data1;

  ap_uint<M_BIT> reg[MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete
  ap_uint<M_BIT*2> data_in;
  ap_uint<M_BIT*2> data_acc;
  ap_uint<M_BIT*2> res_out;




  ap_uint<2*M_BIT> row_buf[MAX_OUP][MAX_W];
#pragma HLS ARRAY_PARTITION variable=row_buf dim=1 complete

  // ap_uint<2*M_BIT> temp_2m;
  // ap_uint<2*M_BIT> res_2m;
  ap_int<M_BIT> temp0,temp1;
  ap_int<M_BIT> res0,res1;
  ap_int<M_BIT> res_buf0,res_buf1;


  unsigned int w=0;
  unsigned int infoldIdx=0;
  unsigned int outfoldIdx=0;
for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL
  (data1, data0) = fifo_C_in[i].read();
  reg[i]=data1;
}

//  for(unsigned int i=0;i<MAX_W;i++){
//#pragma HLS PIPELINE II=1
//    for(unsigned int j=0; j< MAX_OUP;j++){
//    #pragma HLS UNROLL
//        row_buf[j][i]=0;
//    }
//  }


  for (unsigned int h = 0; h < total_num; h++) { // 40
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE false inter variable=row_buf

      for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL  

        (data1, data0) = fifo_C_in[i].read();
        data_in=(data0,reg[i]);
        reg[i]=data1;

        (temp1,temp0)=data_in;

        if(infoldIdx==0){
          data_acc=0;
        }
        else{
          data_acc=row_buf[i][w];
        }

        (res1,res0)=data_acc;


        res_buf0=res0+temp0;
        res_buf1=res1+temp1;

        res_out=(res_buf1,res_buf0);

        if(infoldIdx==SIMDNUM*K-1){

          // cout<<"out0:" <<res_buf0<<endl;
          // cout<<"out1:" <<res_buf1<<endl;

          // if(res_buf0!=0||res_buf1!=0){
          //   cout<<"debug"<<endl;
          // }
    
          fifo_C_out[i].write(res_out);
        }

        row_buf[i][w]=res_out;
      }

    if(w==OUT_W/2-1){
      w=0;
      if(infoldIdx==SIMDNUM*K-1){
        infoldIdx=0;
        if(outfoldIdx==PENUM-1){
          outfoldIdx=0;
        }
        else{
          outfoldIdx++;
        }
      }
      else{
        infoldIdx++;
      }
    }
    else{
      w++;
    }
  }
}



/**
 * LN function-form LNU_v9_mm
*/

template <unsigned ILN_WIDTH>
ap_uint<12> compute_mean_var(ap_uint<ILN_WIDTH> temp_x){
	
	ap_uint<12> var_x;
	// dynamic compress
	ap_uint<1> s;
	ap_uint<4> x0_4b;
	ap_uint<12> x0_12b;
	ap_uint<2> contrl_reg;
  ap_uint<2> shift_value;

	contrl_reg=temp_x[7,6];
	
	if((contrl_reg!=0)){
		s=1;
    shift_value=4;
	}
	else{
		s=0;
    shift_value=2;
	}
  temp_x=temp_x>>shift_value;

	x0_4b=temp_x(3,0);
	#ifdef RESULT_DEBUG
		cout<<"x0_4b: "<<x0_4b<<endl;
		cout<<"shift: "<<(4*s)<<endl;
	#endif

	var_x=(ap_uint<12>(x0_4b*x0_4b))<<(4*s);

	#ifdef RESULT_DEBUG
		cout<<"var_x: "<<var_x<<endl;
	#endif

	return var_x;
	
}





template <unsigned IN_WIDTH,unsigned IN_INTEGER_WIDTH,
          unsigned OUT_WIDTH,unsigned OUT_INTEGER_WIDTH>
ap_fixed<OUT_WIDTH,OUT_INTEGER_WIDTH> compute_silu(ap_fixed<IN_WIDTH,IN_INTEGER_WIDTH> input){

	const ap_fixed<16, 2> onedivsixth = 0.16666666; 

  #ifdef SILU_DEBUG
	  cout<<"input: "<<input<<endl;
  #endif
	// x+3
	ap_fixed<IN_WIDTH,IN_INTEGER_WIDTH> temp=input+3;

  #ifdef SILU_DEBUG
	  cout<<"temp: "<<temp<<endl;
  #endif

	ap_fixed<16,4> relu6_temp;
	if(temp>=6){
		relu6_temp=6;
	}
	else if(temp<=0){
		relu6_temp=0;
	}
	else{
		relu6_temp=temp;
	}

  #ifdef SILU_DEBUG
	  cout<<"relu6_temp: "<<relu6_temp<<endl;
	#endif

	ap_fixed<16,4> temp_div_6;
	temp_div_6=relu6_temp*onedivsixth;

  #ifdef SILU_DEBUG
	  cout<<"temp_div_6: "<<temp_div_6<<endl;
  #endif

	ap_fixed<OUT_WIDTH,OUT_INTEGER_WIDTH> silu_out;

	silu_out=temp_div_6*input;

  #ifdef SILU_DEBUG
	  cout<<"silu_out: "<<silu_out<<endl;
  #endif


	return silu_out;
	
}



template <unsigned IN_WIDTH,unsigned IN_INTEGER_WIDTH,
          unsigned OUT_WIDTH,unsigned OUT_INTEGER_WIDTH>
ap_fixed<OUT_WIDTH,OUT_INTEGER_WIDTH> compute_gelu(ap_fixed<IN_WIDTH,IN_INTEGER_WIDTH> input){
#pragma HLS INLINE OFF
	const ap_fixed<16, 2> onedivsixth = 0.16666666; 
    const ap_fixed<16, 2> constant_factor= 1.702; 

  #ifdef GELU_DEBUG
	  cout<<"input: "<<input<<endl;
  #endif
    // x=x*1.702
    ap_fixed<IN_WIDTH,IN_INTEGER_WIDTH> temp0=input*constant_factor;

  #ifdef GELU_DEBUG
	  cout<<"temp0: "<<temp0<<endl;
  #endif

	// x+3
	ap_fixed<IN_WIDTH,IN_INTEGER_WIDTH> temp=temp0+3;

  #ifdef GELU_DEBUG
	  cout<<"temp: "<<temp<<endl;
  #endif

	ap_fixed<16,4> relu6_temp;
	if(temp>=6){
		relu6_temp=6;
	}
	else if(temp<=0){
		relu6_temp=0;
	}
	else{
		relu6_temp=temp;
	}

  #ifdef GELU_DEBUG
	  cout<<"relu6_temp: "<<relu6_temp<<endl;
	#endif

	ap_fixed<16,4> temp_div_6;
	temp_div_6=relu6_temp*onedivsixth;

  #ifdef GELU_DEBUG
	  cout<<"temp_div_6: "<<temp_div_6<<endl;
  #endif

	ap_fixed<OUT_WIDTH,OUT_INTEGER_WIDTH> silu_out;

	silu_out=temp_div_6*input;

  #ifdef GELU_DEBUG
	  cout<<"silu_out: "<<silu_out<<endl;
  #endif


	return silu_out;
	
}


// template <unsigned ILN_OUT_WIDTH,unsigned ILN_OUT_INTEGER_WIDTH, unsigned SILU_BIT,unsigned SILU_INTEGER_BIT,
//           unsigned MAX_NORM_PE>
// void SiLU_Unit(stream<ap_uint<ILN_OUT_WIDTH * 2> > in[MAX_NORM_PE],
// 	stream<ap_uint<SILU_BIT * 2> > out[MAX_NORM_PE],
// 	const unsigned NumLines,
//   const bool SA_MODE,
// 	const bool NORM_MODE,
//   const bool QUAN_MODE
// 	){

// 	if(QUAN_MODE==false){
// 		return;
// 	}

// 	ap_uint<ILN_OUT_WIDTH> x0,x1;
// 	ap_fixed<ILN_OUT_WIDTH,ILN_OUT_INTEGER_WIDTH> fixp_x0,fixp_x1;

// 	ap_fixed<SILU_BIT,SILU_INTEGER_BIT> out0,out1;
// 	ap_uint<2*SILU_BIT> res;

// 	const ap_fixed<16, 2> onedivsixth = 0.16666666; 


// 	for (unsigned i = 0; i < NumLines; i++) {
// #pragma HLS PIPELINE II=1
//       for(unsigned int c = 0; c < MAX_NORM_PE; c++){
//         ap_uint<ILN_OUT_WIDTH * 2> temp = in[c].read();

//         if(NORM_MODE==true&&SA_MODE==true){
//           (x1,x0)=temp;
//           fixp_x0(ILN_OUT_WIDTH-1,0)=x0(ILN_OUT_WIDTH-1,0);
//           fixp_x1(ILN_OUT_WIDTH-1,0)=x1(ILN_OUT_WIDTH-1,0);
//           out0=compute_silu<ILN_OUT_WIDTH,ILN_OUT_INTEGER_WIDTH,SILU_BIT,SILU_INTEGER_BIT>(fixp_x0);
//           out1=compute_silu<ILN_OUT_WIDTH,ILN_OUT_INTEGER_WIDTH,SILU_BIT,SILU_INTEGER_BIT>(fixp_x1);

//           res=(out1(SILU_BIT-1,0),out0(SILU_BIT-1,0));

//         }
//         else{
//           res=temp;

//         }
//         out[c].write(res);
// 	  }
// 	}

// }



template <	unsigned Wbit,
			unsigned Ibit,
			unsigned Mbit,
			unsigned PACKNUM,
			unsigned P>
ap_int<Mbit> DOT_NPacking(
	ap_uint<P*PACKNUM*Wbit> weights, 
	ap_uint<P*PACKNUM*Ibit> in) 
{	
	ap_int<Mbit> accumulation = 0;

	for (unsigned p = 0; p < P; p++) {
#pragma HLS UNROLL
		ap_int<Mbit> result;
		ap_uint<Wbit> W1,W0;
		ap_uint<Ibit> A1,A0; 

		(W1,W0)= weights( (p+1)*PACKNUM*Wbit-1, p*PACKNUM*Wbit );
		(A1,A0) = in( (p+1)*PACKNUM*Ibit-1, p*PACKNUM*Ibit );

		// cout<<"W1:"<<ap_int<Wbit>(W1)<<"  W0:"<<ap_int<Wbit>(W0)<<endl;
		// cout<<"A1:"<<ap_int<Ibit>(A1)<<"  A0:"<<ap_int<Ibit>(A0)<<endl;
		ap_int<13> test=(ap_int<Wbit>(W1)*ap_int<Ibit>(A1))+(ap_int<Wbit>(W0)*ap_int<Ibit>(A0));

		ap_uint<18> B_port=((ap_uint<18>)W1<<14)+W0;
		ap_uint<22> A_port=((ap_uint<22>)A0<<14);
		ap_uint<Ibit> D_port=A1;
		ap_uint<42> P_port=(A_port+D_port)*B_port;
		


		ap_uint<1> signw0=W0.range(Wbit-1,Wbit-1);
		ap_uint<1> signw1=W1.range(Wbit-1,Wbit-1);
		ap_uint<1> signa0=A0.range(Ibit-1,Ibit-1);
		ap_uint<1> signa1=A1.range(Ibit-1,Ibit-1);

		ap_uint<8> sign8w0=(signw0,signw0,signw0,signw0,signw0,signw0,signw0,signw0);
		ap_uint<8> sign8w1=(signw1,signw1,signw1,signw1,signw1,signw1,signw1,signw1);
		ap_uint<4> sign4a0=(signa0,signa0,signa0,signa0);
		ap_uint<4> sign4a1=(signa1,signa1,signa1,signa1);

		ap_uint<9> out0_L0=(sign8w0&A0)+(sign8w1&A1);
		ap_uint<5> out0_L1=(sign4a0&W0)+(sign4a1&W1);
		ap_uint<12> res_correct=(ap_uint<13>(out0_L0)<<4)+(ap_uint<13>(out0_L1)<<8);
		
		ap_uint<13> out=P_port(26,14)-res_correct;
		// cout<<"out:"<<out<<endl;
		ap_int<12> result_correct=out;
		// cout<<"result_correct:"<<result_correct<<endl;
//		if(test!=result_correct){
//			cout<<"test error....!"<<endl;
//		}

		accumulation += result_correct;
	}

	return accumulation;
}




template <	unsigned DEQUAN_BIT,
      unsigned MAX_INP,
			unsigned MAX_OUP,
			unsigned MAX_SOFTMAX_INBUF_LENGTH>
void SOFTMAX_WriteBUF(stream<ap_uint<DEQUAN_BIT*2> > in[MAX_OUP], 
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_SOFTMAX_INBUF_LENGTH],
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


template <	unsigned DEQUAN_BIT,
      unsigned DEQUAN_INTEGER_BIT,
			unsigned MAX_OUP>
void FIND_MAX_VALUE(
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> OUP_TempBuf[MAX_OUP],
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> &MAX_Temp
	){
//#pragma HLS INLINE OFF
//#pragma HLS latency max=1


	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp[MAX_OUP/2];

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp_temp[2];

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp_temp_temp;

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp_temp_temp_temp;

	for(unsigned i=0; i<MAX_OUP/2;i++){  // 8
#pragma HLS UNROLL
		temp[i]=OUP_TempBuf[2*i]>OUP_TempBuf[2*i+1]?OUP_TempBuf[2*i]:OUP_TempBuf[2*i+1];
	}

  temp_temp[0]=temp[0]>temp[1]?temp[0]:temp[1];
  temp_temp[1]=temp[2]>temp[3]?temp[2]:temp[3];

  temp_temp_temp=temp_temp[0]>temp_temp[1]?temp_temp[0]:temp_temp[1];
  temp_temp_temp_temp=temp_temp_temp>temp[4]?temp_temp_temp:temp[4];


	MAX_Temp=MAX_Temp>temp_temp_temp_temp?MAX_Temp:temp_temp_temp_temp;

}


template <	unsigned DEQUAN_BIT,
      unsigned DEQUAN_INTEGER_BIT,
      unsigned MAX_INP,
			unsigned MAX_OUP,
			unsigned MAX_SOFTMAX_INBUF_LENGTH>
void SOFTMAX_WriteBUF_ADDBUF(stream<ap_uint<DEQUAN_BIT*2> > in[MAX_OUP], 
	ap_uint<DEQUAN_BIT*2> ROW_T_buf[MAX_OUP][MAX_SOFTMAX_INBUF_LENGTH],
  stream<ap_uint<DEQUAN_BIT*2> > NO_SOFTMAX_OUT[MAX_OUP],
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> tmax_M[MAX_INP][2],
	const unsigned PENUM,
  const bool EBMULT_MODE
	){
#pragma HLS INLINE OFF

	// unsigned int loop0,loop1,loop2;
	unsigned int numlines;

	unsigned int outdIdx=0;
    unsigned int w=0;
    unsigned int h=0;
    unsigned int index;

	// loop0=2;
	// loop1=MAX_INP;
	// loop2=PENUM; // M/(MAX_OUP*2)
	numlines= PENUM*MAX_INP*2;
	ap_uint< DEQUAN_BIT*2> temp;
	ap_uint<DEQUAN_BIT> temp_x0, temp_x1;
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> temp_x0_fixp, temp_x1_fixp;

	ap_uint< DEQUAN_BIT*2> first_temp;
	ap_uint<DEQUAN_BIT> first_temp_x0, first_temp_x1;
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> first_temp_x0_fixp, first_temp_x1_fixp;

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> first_temp_out0_fixp, first_temp_out1_fixp;

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> MAX_TempBuf[MAX_INP][2];
#pragma HLS ARRAY_PARTITION variable=MAX_TempBuf complete dim=2

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> OUP_TempBuf[2][MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=OUP_TempBuf complete dim=0

	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> OUP_out_TempBuf[2][MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=OUP_out_TempBuf complete dim=0


	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> max_before_temp0,max_after_temp0;
	ap_fixed<DEQUAN_BIT, DEQUAN_INTEGER_BIT> max_before_temp1,max_after_temp1;


	for(unsigned j=0; j<MAX_INP;j++){
#pragma HLS UNROLL
		MAX_TempBuf[j][0]=-128;
		MAX_TempBuf[j][1]=-128;
	}

	for(unsigned m=0; m<numlines;m++){
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE false inter variable=ROW_T_buf
		index=outdIdx*PENUM*2+h*2+w;
		for(unsigned i=0; i<MAX_OUP;i++){
#pragma HLS UNROLL			
			temp = in[i].read();


			(temp_x1,temp_x0)=temp;

			temp_x0_fixp(DEQUAN_BIT-1,0)=temp_x0(DEQUAN_BIT-1,0); 
			temp_x1_fixp(DEQUAN_BIT-1,0)=temp_x1(DEQUAN_BIT-1,0);

			#ifdef RESULT_DEBUG
				cout<<"temp_x0_fixp:"<<temp_x0_fixp<<endl; // x_i
				cout<<"temp_x1_fixp:"<<temp_x1_fixp<<endl;  // t_i-1
			#endif

			first_temp=ROW_T_buf[index+i];

			(first_temp_x1,first_temp_x0)=first_temp;

			first_temp_x0_fixp(DEQUAN_BIT-1,0)=first_temp_x0(DEQUAN_BIT-1,0); 
			first_temp_x1_fixp(DEQUAN_BIT-1,0)=first_temp_x1(DEQUAN_BIT-1,0);

			#ifdef RESULT_DEBUG
				cout<<"first_temp_x0_fixp:"<<first_temp_x0_fixp<<endl; // x_i
				cout<<"first_temp_x1_fixp:"<<first_temp_x1_fixp<<endl;  // t_i-1
			#endif

			// first_temp_out0_fixp=(temp_x0_fixp<<4)+first_temp_x0_fixp;
			// first_temp_out1_fixp=(temp_x1_fixp<<4)+first_temp_x1_fixp;

			first_temp_out0_fixp=(temp_x0_fixp)+first_temp_x0_fixp;
			first_temp_out1_fixp=(temp_x1_fixp)+first_temp_x1_fixp;

			#ifdef RESULT_DEBUG
				cout<<"add:"<<first_temp_out0_fixp<<endl; // x_i
				cout<<"add:"<<first_temp_out1_fixp<<endl;  // t_i-1
			#endif

			
			OUP_TempBuf[0][i]=first_temp_out0_fixp;
			OUP_TempBuf[1][i]=first_temp_out1_fixp;

      if(EBMULT_MODE){
        NO_SOFTMAX_OUT[i].write((first_temp_out1_fixp(DEQUAN_BIT-1,0),first_temp_out0_fixp(DEQUAN_BIT-1,0)));
      }
      else{
        ROW_T_buf[i][index]=(first_temp_out1_fixp(DEQUAN_BIT-1,0),first_temp_out0_fixp(DEQUAN_BIT-1,0));
      }
			

		}



		FIND_MAX_VALUE<DEQUAN_BIT,DEQUAN_INTEGER_BIT,MAX_OUP>(OUP_TempBuf[0],MAX_TempBuf[outdIdx][0]);
		FIND_MAX_VALUE<DEQUAN_BIT,DEQUAN_INTEGER_BIT,MAX_OUP>(OUP_TempBuf[1],MAX_TempBuf[outdIdx][1]);

		// cout<<"MAX_TempBuf["<<outdIdx<<"][0]"<<MAX_TempBuf[outdIdx][0]<<endl;
		// cout<<"MAX_TempBuf["<<outdIdx<<"][1]"<<MAX_TempBuf[outdIdx][1]<<endl;


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

	for(unsigned j=0; j<MAX_INP;j++){
#pragma HLS UNROLL
		tmax_M[j][0]=MAX_TempBuf[j][0];
		// cout<<tmax_M[j][0]<<endl;
		tmax_M[j][1]=MAX_TempBuf[j][1];
		// cout<<tmax_M[j][1]<<endl;
	}


}







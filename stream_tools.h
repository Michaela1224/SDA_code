#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
using namespace std;

//#define WINPUT_DEBUG

template <unsigned DataWidth>
void DemuxStream2 (
	stream<ap_uint<DataWidth> >& in, 
	stream<ap_uint<DataWidth> >& out1, 
	stream<ap_uint<DataWidth> >& out2, 
	const unsigned mode, 
    const unsigned NumLines)
{
	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		ap_uint<DataWidth> temp = in.read();
		if (mode == 0)
			out1.write(temp);  // to_mm
		else
			out2.write(temp);   // to_conv3
	}
}


template <unsigned DataWidth>
void MuxStream2(
	stream<ap_uint<DataWidth> >& in1, 
	stream<ap_uint<DataWidth> >& in2,
	stream<ap_uint<DataWidth> >& out, 
	const bool mode, 
	const unsigned NumLines)
{
	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		ap_uint<DataWidth> temp;
		if (mode == false)
			temp = in1.read();
		else
			temp = in2.read();
		out.write(temp);
	}
}



template <unsigned DataWidth, unsigned MAX_PE>
void MuxStream2_P(
	stream<ap_uint<DataWidth> > in1[MAX_PE], 
	stream<ap_uint<DataWidth> > in2[MAX_PE],
	stream<ap_uint<DataWidth> > out[MAX_PE], 
	const unsigned NumLines,
	const bool mode)
{


	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		ap_uint<DataWidth> temp;
		for(unsigned j=0; j<MAX_PE; j++){
			if (mode == false)
				temp = in1[j].read();
			else
				temp = in2[j].read();
			out[j].write(temp);
		}

	}
}


template <unsigned DataWidth, unsigned MAX_PE>
void MuxStream2_P_BRANCH(
	stream<ap_uint<DataWidth> > in1[MAX_PE], 
	stream<ap_uint<DataWidth> > in2[MAX_PE],
	stream<ap_uint<DataWidth> > out[MAX_PE], 
	const unsigned NumLines,
	const bool mode,
	const bool mode_en)
{
	if(mode_en==false){
		return;
	}

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		ap_uint<DataWidth> temp;
		for(unsigned j=0; j<MAX_PE; j++){
			if (mode == false)
				temp = in1[j].read();
			else
				temp = in2[j].read();
			out[j].write(temp);
		}

	}
}



template <unsigned DataWidth, unsigned MAX_PE>
void MuxStream3_P_BRANCH(
	stream<ap_uint<DataWidth> > in_shortcut[MAX_PE], 
	stream<ap_uint<DataWidth> > in_nonorm[MAX_PE],
	stream<ap_uint<DataWidth> > in_emult[MAX_PE],
	stream<ap_uint<DataWidth> > out[MAX_PE], 
	const unsigned NumLines,
	const bool SHORCUT_ADD_MODE,
	const bool EBMULT_MODE,
	const bool SHORCUT_QUAN_MODE)
{
	if(SHORCUT_QUAN_MODE==false){
		return;
	}

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		ap_uint<DataWidth> temp;
		for(unsigned j=0; j<MAX_PE; j++){
			if (SHORCUT_ADD_MODE == true){
				temp = in_nonorm[j].read();
			}
			else if(EBMULT_MODE ==true){
				temp = in_emult[j].read();
			}
			else{
				temp = in_shortcut[j].read();
			}

			out[j].write(temp);
		}

	}
}


template <unsigned DataWidth, unsigned MAX_PE>
void MuxStream3_P(
	stream<ap_uint<DataWidth> > norm_in[MAX_PE], 
	stream<ap_uint<DataWidth> > softmax_in[MAX_PE],
	stream<ap_uint<DataWidth> > gelu_in[MAX_PE],
	stream<ap_uint<DataWidth> > out[MAX_PE], 
	const unsigned NumLines,
	const bool NORM_MODE,
	const bool SOFTMAX_MODE,
	const bool GELU_MODE,
	const bool TRANSPOSE_MODE
	)
{

	if(NORM_MODE==false&&SOFTMAX_MODE==false&&GELU_MODE==false&&TRANSPOSE_MODE==false){
		return;
	}

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		ap_uint<DataWidth> temp;
		for(unsigned j=0; j<MAX_PE; j++){
			if (NORM_MODE)
				temp = norm_in[j].read();
			else if (SOFTMAX_MODE)
				temp = softmax_in[j].read();
			else if (GELU_MODE|TRANSPOSE_MODE)
				temp = gelu_in[j].read();
			out[j].write(temp);
		}

	}
}


template <unsigned A_ROW, unsigned A_COL,unsigned DataWidth>
void MuxStream2_RC(
	stream<ap_uint<DataWidth> > in1[A_ROW][A_COL], 
	stream<ap_uint<DataWidth> > in2[A_ROW][A_COL],
	stream<ap_uint<DataWidth> > out[A_ROW][A_COL], 
	const unsigned NumLines,
	const bool mode)
{
	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		for (unsigned x = 0; x < A_ROW; x++) {
			for (unsigned y = 0; y < A_COL; y++) {
				ap_uint<DataWidth> temp;
				if (mode == false)
					temp = in1[x][y].read();
				else
					temp = in2[x][y].read();
				out[x][y].write(temp);
			}
		}
	}
}


template <unsigned MAX_OUP, unsigned PACK_NUM,
		  unsigned PACK_CONV_NUM, unsigned W_BIT>
void MM_to_CONV3_Stream(
	stream<ap_uint<MAX_OUP * W_BIT * PACK_NUM> >& in, 
	stream<ap_uint<MAX_OUP * W_BIT * PACK_CONV_NUM> >& out, 
    const unsigned NumLines,
	const bool skip_mode)
{



    if(skip_mode==true){
      return;
    }

#ifdef WINPUT_DEBUG
    //   FILE* fpa = fopen("a_stream_pe00_gold.txt", "wb");
    FILE* fpw = fopen("w_stream.txt", "wb");
#endif

	ap_uint<MAX_OUP * W_BIT * PACK_NUM> temp_in;
	ap_uint<W_BIT> temp_w0;
	ap_uint<W_BIT> temp_w1;
	ap_uint<W_BIT * PACK_CONV_NUM> temp_w_exp;
	ap_uint<MAX_OUP * W_BIT * PACK_CONV_NUM> temp_out;

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1 
		temp_in = in.read();
		
		for(unsigned j=0;j<MAX_OUP;j++){
			(temp_w1,temp_w0)=temp_in((j+1)*W_BIT * PACK_NUM-1,j*W_BIT * PACK_NUM);

			#ifdef WINPUT_DEBUG
				ap_int<W_BIT> test_w0=temp_w0;
				ap_int<W_BIT> test_w1=temp_w1;
				fprintf(fpw, "%d\n", (int)test_w0);
				fprintf(fpw, "%d\n", (int)test_w1);
			#endif
			temp_out=temp_out>>(W_BIT*PACK_CONV_NUM);
			temp_w_exp=((ap_uint<PACK_CONV_NUM * W_BIT>)temp_w1<<8)+temp_w0;
			temp_out(MAX_OUP * PACK_CONV_NUM * W_BIT-1,((MAX_OUP-1) * PACK_CONV_NUM) * W_BIT)=temp_w_exp;
		}
		out.write(temp_out);

	}

#ifdef WINPUT_DEBUG
	fclose(fpw);
#endif

}



template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned MAX_PE>
void ExpandWidth_P(
	stream<ap_uint<InStreamW> > in[MAX_PE],
	stream<ap_uint<OutStreamW> > out[MAX_PE],
	const unsigned NumLines,
	const unsigned skip_mode
	){

    if(skip_mode==0){
      return;
    }

	const unsigned parts = OutStreamW/InStreamW;
	ap_uint<OutStreamW> buffer[MAX_PE];
#pragma HLS ARRAY_PARTITION variable=temp_in dim=1 complete	
	int index=0;


	for (unsigned rep = 0; rep < NumLines*parts; rep++) {  //400*400
#pragma HLS loop_tripcount min=NumLines max=NumLines

#pragma HLS PIPELINE II=1
		for (unsigned i = 0; i < MAX_PE; i++) {
			ap_uint<InStreamW> temp = in[i].read();
			buffer[i]( (index+1)*InStreamW-1, index*InStreamW ) = temp;
		}

		if(index==parts-1){
			for (unsigned i = 0; i < MAX_PE; i++) {
				out[i].write(buffer[i]);
			}
		}



		if(index==parts-1){
			index=0;
		}
		else{
			index++;
		}

	}


}




template <	unsigned InStreamW,
			unsigned IN_PE,
			unsigned OU_PE>
void ExpandWidth_OUP(
	stream<ap_uint<InStreamW> > in[IN_PE],
	stream<ap_uint<InStreamW> > out[OU_PE],
	const unsigned NumLines,
	const bool skip_mode
	){

    if(skip_mode==false){
      return;
    }

	const unsigned parts = OU_PE/IN_PE;

	int index=0;


	for (unsigned rep = 0; rep < NumLines; rep++) {  //400*400
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=1

		for (unsigned i = 0; i < IN_PE; i++) {
			ap_uint<InStreamW> temp = in[i].read();
			out[index*IN_PE+i].write(temp);
		}

		if(index==parts-1){
			index=0;
		}
		else{
			index++;
		}

	}


}



template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned MAX_PE>
void ReduceWidth_P(
	stream<ap_uint<InStreamW> > in[MAX_PE],
	stream<ap_uint<OutStreamW> > out[MAX_PE],
	const unsigned NumLines,
	const bool skip_mode){

    if(skip_mode==true){
      return;
    }

	const unsigned parts = InStreamW/OutStreamW;
	ap_uint<InStreamW> temp_in[MAX_PE];
#pragma HLS ARRAY_PARTITION variable=temp_in dim=1 complete	

	for (unsigned rep = 0; rep < NumLines; rep++) {  //400*400*3*3
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=1
		for (unsigned i = 0; i < MAX_PE; i++) {
			temp_in[i] = in[i].read();
			for (unsigned p = 0; p < parts; p++) {

				ap_uint<OutStreamW> temp_out = temp_in[i](OutStreamW-1, 0);
				out[i].write(temp_out);
				temp_in[i] = temp_in[i] >> OutStreamW;
			}
		}
	}
}





template <	unsigned InStreamW,
			unsigned OutStreamW>
void ReduceWidth(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned NumLines)
{
	const unsigned parts = InStreamW/OutStreamW;

	for (unsigned rep = 0; rep < NumLines; rep++) {  //400*400*3*3
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=InStreamW/OutStreamW

		ap_uint<InStreamW> temp_in = in.read();
		for (unsigned p = 0; p < parts; p++) {

			ap_uint<OutStreamW> temp_out = temp_in(OutStreamW-1, 0);
			out.write( temp_out );
			temp_in = temp_in >> OutStreamW;
		}
	}
}


template <	unsigned InStreamW,
			unsigned OutStreamW>
void ReduceWidth_EN(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned NumLines,
	const bool skip_mode)
{
    if(skip_mode==false){
      return;
    }
	const unsigned parts = InStreamW/OutStreamW;

	for (unsigned rep = 0; rep < NumLines; rep++) {  //400*400*3*3
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=InStreamW/OutStreamW

		ap_uint<InStreamW> temp_in = in.read();
		for (unsigned p = 0; p < parts; p++) {

			ap_uint<OutStreamW> temp_out = temp_in(OutStreamW-1, 0);
			out.write( temp_out );
			temp_in = temp_in >> OutStreamW;
		}
	}
}
############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project hls_prj
set_top do_compute_top
add_files stream_tools.h
add_files sa_tools.h
add_files param_sa.h
add_files load_param.h
add_files diffusion-lib.h
add_files config.h
add_files block_top.h
add_files block_top.cpp
add_files -tb block_test.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb config_test.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb mm_bias_160.bin -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb mm_fm_trans_160.bin -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb mm_w_160.bin -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb test.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution3_d128_gelu" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 3.3 -name default
config_export -format ip_catalog -rtl verilog
source "./hls_prj/solution3_d128_gelu/directives.tcl"
csim_design -clean -setup
csynth_design
cosim_design -trace_level all
export_design -rtl verilog -format ip_catalog

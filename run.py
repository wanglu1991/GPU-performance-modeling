#!/usr/bin/env python 
import os
import sys
def main(kernel_numbers):
    for i in range(1, int(kernel_numbers)+1):
          os.system('python ./generate_L2_access.py '+str(i))
          os.system('python ./interval_warp_new.py '+str(i)+' > out_IPC_kernel_'+str(i))
          os.system('rm ./read_out_*')
          os.system('rm ./output_*')
    result=open('./result_delay_model','r')
    accum_cycle=0
    accum_inst=0
    for line in result:
        ipc=float(line.split(',')[0])
        instruction_count=float(line.split(',')[1])
        cycle=instruction_count/ipc
        accum_cycle=accum_cycle+cycle
        accum_inst=accum_inst+instruction_count
    total_ipc=float(accum_inst)/float(accum_cycle)
    print('ipc_delay_model')
    print(total_ipc)
    result=open('./result_delay_model+optimal_MSHR','r')
    accum_cycle=0
    accum_inst=0
    for line in result:
        ipc=float(line.split(',')[0])
        instruction_count=float(line.split(',')[1])
        cycle=instruction_count/ipc
        accum_cycle=accum_cycle+cycle
        accum_inst=accum_inst+instruction_count
    total_ipc=float(accum_inst)/float(accum_cycle)
    print('ipc_delay_model+optimal_MSHR')
    print(total_ipc)
    result=open('./result_delay_model+ceiling_MSHR','r')
    accum_cycle=0
    accum_inst=0
    for line in result:
        ipc=float(line.split(',')[0])
        instruction_count=float(line.split(',')[1])
        cycle=instruction_count/ipc
        accum_cycle=accum_cycle+cycle
        accum_inst=accum_inst+instruction_count
    total_ipc=float(accum_inst)/float(accum_cycle)
    print('ipc_delay_model+ceiling_MSHR')
    print(total_ipc)
if __name__== '__main__':
    main(sys.argv[1])

#!/usr/bin/env python 
import os
import sys
def main(kernel_numbers):
 #   for i in range(1, int(kernel_numbers)+1):
 #       os.system('python ./generate_L2_access.py '+str(i))
 #       os.system('python ./interval_warp.py '+str(i)+' > out_IPC_kernel_'+str(i))
 #       os.system('rm ./read_out_*')
 #       os.system('rm ./output_*')
    result=open('./base_result.txt','r')
    MSHR_result=open('./MSHR_delay_result.txt','w')
    MSHR_ideal_result=open('./MSHR_ideal_result.txt','w')
    MSHR_DRAM_result=open('./MSHR_DRAM_result.txt','w')
    MSHR_DRAM_ideal_result=open('./MSHR_DRAM_ideal_result.txt','w')
    multi_result=open('./multi_result.txt','w')
    accum_cycle=0
    accum_inst=0
  #  for line in result:
  #      base_ipc=float(line.split(',')[2])
  #      instruction_count=float(line.split(',')[-1])
  #      cycle=instruction_count/base_ipc
  #      accum_cycle=accum_cycle+cycle
  #      accum_inst=accum_inst+instruction_count
  #  total_base_ipc=float(accum_inst)/float(accum_cycle)
  #  print(total_base_ipc)
    for line in result:
        base_cycle=float(line.split(',')[1])
        MSHR_delay=float(line.split(',')[3])
        multi_result.write(str(line.split(',')[2])+','+str(line.split(',')[-1]))
        total_MSHR_cycle=base_cycle+MSHR_delay
        total_MSHR_DRAM_cycle=total_MSHR_cycle+float(line.split(',')[4])
        total_ideal_cycle=base_cycle+70
        total_MSHR_ipc=float(line.split(',')[0])/float(total_MSHR_cycle)
        MSHR_result.write(str(total_MSHR_ipc)+','+str(line.split(',')[-1]))
        MSHR_ideal_cycle=base_cycle+8825+float(line.split(',')[4])
        MSHR_ideal_ipc=float(line.split(',')[0])/float(MSHR_ideal_cycle)
        MSHR_ideal_result.write(str(MSHR_ideal_ipc)+','+str(line.split(',')[-1]))
        total_MSHR_DRAM_ipc=float(line.split(',')[0])/float(total_MSHR_DRAM_cycle)
        total_ideal_ipc=float(line.split(',')[0])/float(total_ideal_cycle)
        MSHR_DRAM_result.write(str(total_MSHR_DRAM_ipc)+','+str(line.split(',')[-1]))
        MSHR_DRAM_ideal_result.write(str(total_ideal_ipc)+','+str(line.split(',')[-1]))
if __name__== '__main__':
    main(sys.argv[1])

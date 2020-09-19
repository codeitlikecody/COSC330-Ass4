################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../mandelbrotCUDA.cu 

C_SRCS += \
../bmpfile.c 

O_SRCS += \
../bmpfile.o \
../mandelbrotCUDA.o 

OBJS += \
./bmpfile.o \
./mandelbrotCUDA.o 

CU_DEPS += \
./mandelbrotCUDA.d 

C_DEPS += \
./bmpfile.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 -gencode arch=compute_70,code=sm_70  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 -gencode arch=compute_70,code=sm_70  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_70,code=compute_70 -gencode arch=compute_70,code=sm_70  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



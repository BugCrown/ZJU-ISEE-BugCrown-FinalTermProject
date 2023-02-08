 // demo.c
 #include <stdio.h>
 #include <tvm_runtime.h>
 #include "uart.h"
 #include <tvmgen_default.h>
 #include <time.h>

 // Header files generated by convert_image.py and convert_labels.py
 #include "inputs.h"
 #include "labels.h"
 #include "outputs.h"

 int abs(int v) { return v * ((v > 0) - (v < 0)); }

 int main(int argc, char** argv) {
    clock_t start, end;    
    start = clock();   
    uart_init();
   printf("Starting Demo\n");
  //  EthosuInit();

   printf("Running inference\n");

    struct tvmgen_default_outputs outputs = {
        .Identity = output,
    };
    struct tvmgen_default_inputs inputs = {
        .input_1 = input,
    };


    tvmgen_default_run(&inputs, &outputs);

    // Calculate index of max value
    float max_value = -1;
    int32_t max_index = 9;

    for (unsigned int i = 0; i < output_len; ++i) {
      if (output[i] > max_value) {
        max_value = output[i];
        max_index = i;
            
      }
      //  printf("%d %d\n", i, output[i]); 
    }
    printf("The image has been classified as '%s'\n", labels[max_index]);

   // The FVP will shut down when it receives "EXITTHESIM" on the UART

   end = clock();
   printf("end = %f", end);
   double total = end - start;

   printf("%f ms for one prediction\n", total);
   printf("EXITTHESIM\n");
   while (1 == 1)
     ;
   return 0;
 }
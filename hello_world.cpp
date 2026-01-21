/*
 * Author: Jan Eberhardt
 */

#include "global.h"



// using the hello_world_circ_buffer to verify whether the hardware setup is working correctly
CircularBuffer rx_buffer;
CircularBuffer tx_buffer;


// the following arrays/buffers are required in order to loop the data from the input to the output
uint32_t in[BLOCK_SIZE];
uint32_t out[BLOCK_SIZE];
int16_t left_in[BLOCK_SIZE];
int16_t right_in[BLOCK_SIZE];
int16_t left_out[BLOCK_SIZE];
int16_t right_out[BLOCK_SIZE];

float32_t Coeffes[24]= {0};
float32_t* Coeffes_p = Coeffes;
float32_t out_f[BLOCK_SIZE];
float32_t* p_out_f = out_f;
float32_t in_f[BLOCK_SIZE];
float32_t* p_in_f = in_f;
float32_t Sin [256];
float32_t* Sin_p = Sin;



int main()
{
    // initialze whole platform, does not start DMA
    init_platform(115200, hz8000, line_in);

    // use debug_printf() to send data to a Serial Monitor
    debug_printf("%s, %s\n", __DATE__, __TIME__);

    // function calls surrounded by IF_DEBUG() will be removed when building a Release
    IF_DEBUG(debug_printf("Hello World!\n"));

    // init test pin P10 to LOW; can be found on the board as part of the connector CN10, Pin is labelled as A3
    gpio_set(TEST_PIN, LOW);

    // initialize circular buffers
    rx_buffer.init();
    tx_buffer.init();

    memset(in, 0, sizeof(in));
    memset(out, 0, sizeof(out));
    //gen- sin lokk up
    quaterLUT(Sin_p);


    // start I2S, call just before your main loop
    // this command starts the DMA, which will begin transferring data to and from the rx_buffer and tx_buffer
    platform_start();
    while(true)
    {
    // step 1: read block of samples from input buffer
    while(!rx_buffer.read(in));
    
    gpio_set(LED_B, HIGH); // LED_B off
    gpio_set(TEST_PIN, HIGH); // Test Pin High
    
    // step 2: split samples into two channels
    convert_audio_sample_to_2ch(in, left_in, right_in);
    
    // step 3: process the audio channels
    // 3.1: convert from int16_t to float (KEINE Normalisierung!)
    for (uint32_t x = 0; x < BLOCK_SIZE; x++) {
        in_f[x] = (float32_t)left_in[x];  // Direkte Konvertierung
    }
    
    // 3.2: process data mit KLEINEM mu
    blockLMS(Sin_p, p_in_f, p_out_f, Coeffes_p, BLOCK_SIZE, 24, 0.001, 50, 8000);
    //                                                           ^^^^^^^ Sehr kleiner Wert!
    
    // 3.3: convert from float to int16_t
    for (uint32_t x = 0; x < BLOCK_SIZE; x++) {
        // Clipping für Sicherheit
        float temp = out_f[x];
        if (temp > 32767.0f) temp = 32767.0f;
        if (temp < -32768.0f) temp = -32768.0f;
        left_out[x] = (int16_t)temp;
        right_out[x] = left_out[x];  // Mono auf beide Kanäle
    }
    
    // step 4: merge two channels into one sample
    convert_2ch_to_audio_sample(left_out, right_out, out);
    
    // step 5: write block of samples to output buffer
    while(!tx_buffer.write(out));
    
    gpio_set(LED_B, LOW); // LED_B on
    gpio_set(TEST_PIN, LOW); // Test Pin Low
}
    while(false)
    {
        // step 1: read block of samples from input buffer, data is copied from rx_buffer to in
        while(!rx_buffer.read(in));

        // blue LED is used to visualize (processing time)/(sample time)
        gpio_set(LED_B, HIGH);			// LED_B off
        gpio_set(TEST_PIN, HIGH);       // Test Pin High

        // step 2: split samples into two channels
        //convert_audio_sample_to_2ch(in, left_in, right_in);


        // step 3: process the audio channels
        //      3.1: convert from int to float if necessary, see CMSIS_DSP
        //      3.2: process data
        //      3.3: convert from float to int if necessary, see CMSIS_DSP
        

        // step 2: split samples into two channels
        convert_audio_sample_to_2ch(in, left_in, right_in);

// step 3: process the audio channels
// 3.1: convert from int16_t to float und normalisieren
for (uint32_t x = 0; x < BLOCK_SIZE; x++) {
    in_f[x] = (float32_t)left_in[x] / 32768.0f;  // Normalisierung auf [-1, +1]
}

// Nach Filterung zurückskalieren
for (uint32_t x = 0; x < BLOCK_SIZE; x++) {
    float temp = out_f[x] * 32768.0f;
    // Clipping
    if (temp > 32767.0f) temp = 32767.0f;
    if (temp < -32768.0f) temp = -32768.0f;
    left_out[x] = (int16_t)temp;
    right_out[x] = left_out[x];
}
    // 3.2: process data
    blockLMS(Sin_p, p_in_f, p_out_f, Coeffes_p, BLOCK_SIZE, 24, 0.05, 50, 8000);

    // 3.3: convert from float to int16_t
    for (uint32_t x = 0; x < BLOCK_SIZE; x++) {
        left_out[x] = (int16_t)out_f[x];   // Korrekte Konvertierung zu signed int
        right_out[x] = left_out[x];        // Mono auf beide Kanäle
    }

    // step 4: merge two channels into one sample
    convert_2ch_to_audio_sample(left_out, right_out, out);


     /*   for (uint32_t x = 0; x < BLOCK_SIZE; x++)
        {
            in_f[x] = (float32_t)in[x];
        };
    
        blockLMS(Sin_p,p_in_f,p_out_f,Coeffes_p,BLOCK_SIZE, 24, 0.05f ,50 ,8000);

        //float64_t phasenIncrement = 2*3.14 * 50 / 32000;
    
        for (uint32_t x = 0; x < BLOCK_SIZE; x++)
        {

            out[x] = (uint32_t)abs(out_f[x]);//(int)out_f[x];
        };
        */

        // step 4: merge two channels into one sample
        //convert_2ch_to_audio_sample(left_out, right_out, out);

        // step 5: write block of samples to output buffer, data is copied from out to tx_buffer
        while(!tx_buffer.write(out));
        IF_DEBUG(debug_printf("Hello World!0\n"));
        gpio_set(LED_B, LOW);			// LED_B on
        gpio_set(TEST_PIN, LOW);        // Test Pin Low
    }
    IF_DEBUG(debug_printf("Hello World!4\n"));


    // fail-safe, never return from main on a microcontroller
    fatal_error();

    return 0;
}



// the following functions are called, when the DMA has finished transferring one block of samples and needs a new memory address to write/read to/from

// prototype defined in platform.h
// get new memory address to read new data to send it to DAC
uint32_t* get_new_tx_buffer_ptr()
{
    uint32_t* temp = tx_buffer.get_read_ptr();
    if(temp == nullptr)
    {
        fatal_error();
    }
    return temp;
}

// prototype defined in platform.h
// get new memory address to write new data received from ADC
uint32_t* get_new_rx_buffer_ptr()
{
    uint32_t* temp = rx_buffer.get_write_ptr();
    if(temp == nullptr)
    {
        fatal_error();
    }
    return temp;
}


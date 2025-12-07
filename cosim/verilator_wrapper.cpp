#include <verilated.h>
#include "Vreram_controller_verilator.h"
#include <cstdint>
#include <cstdio>
#include <vector>

class ReRAMHardware {
private:
    Vreram_controller_verilator* dut;
    uint64_t sim_time;
    int16_t (*xbar_callback)(uint16_t neuron_idx, const uint8_t* pixels);
    
    // Debug counters
    int debug_xbar_requests;
    int debug_xbar_responses;

public:
    ReRAMHardware() {
        dut = new Vreram_controller_verilator;
        sim_time = 0;
        xbar_callback = nullptr;
        debug_xbar_requests = 0;
        debug_xbar_responses = 0;

        // Initialize all signals
        dut->clk = 0;
        dut->rst_n = 0;
        dut->start = 0;
        dut->input_pixel = 0;
        dut->input_valid = 0;
        dut->xbar_result = 0;
        dut->xbar_valid = 0;
        
        dut->eval();
    }

    ~ReRAMHardware() {
        dut->final();
        delete dut;
    }

    void set_crossbar_callback(int16_t (*callback)(uint16_t, const uint8_t*)) {
        xbar_callback = callback;
    }

    void clock_tick() {
        // Rising edge
        dut->clk = 1;
        dut->eval();
        sim_time++;

        // Falling edge
        dut->clk = 0;
        dut->eval();
        sim_time++;
    }

    void reset() {
        dut->rst_n = 0;
        for (int i = 0; i < 5; i++) {
            clock_tick();
        }
        dut->rst_n = 1;
        clock_tick();
        
        // Reset debug counters
        debug_xbar_requests = 0;
        debug_xbar_responses = 0;
    }

    int run_inference(const uint8_t* input_pixels, int16_t* output_neurons) {
        // Initialize output buffer to zero
        for (int i = 0; i < 256; i++) {
            output_neurons[i] = 0;
        }
        
        reset();

        // Start inference
        dut->start = 1;
        clock_tick();
        dut->start = 0;
        clock_tick();

        // Wait for busy to assert
        int timeout_busy = 0;
        while (!dut->busy && timeout_busy < 100) {
            clock_tick();
            timeout_busy++;
        }

        if (!dut->busy) {
            fprintf(stderr, "ERROR: Hardware did not start (busy not asserted)\n");
            return -1;
        }

        // ========== PIXEL LOADING ==========
        int pixel_idx = 0;
        
        fprintf(stderr, "DEBUG: Starting pixel load...\n");
        
        while (pixel_idx < 784) {
            if (dut->input_ready) {
                dut->input_pixel = input_pixels[pixel_idx];
                dut->input_valid = 1;
                clock_tick();
                dut->input_valid = 0;
                pixel_idx++;
                
                // Debug every 100 pixels
                if (pixel_idx % 100 == 0) {
                    fprintf(stderr, "DEBUG: Loaded %d pixels\n", pixel_idx);
                }
            } else {
                clock_tick();
            }
        }
        
        fprintf(stderr, "DEBUG: Pixel loading complete - loaded %d pixels\n", pixel_idx);

        if (pixel_idx < 784) {
            fprintf(stderr, "ERROR: Only loaded %d/784 pixels\n", pixel_idx);
            return -2;
        }
        
        // ========== NEURON PROCESSING ==========
        int neuron_count = 0;
        const int MAX_CYCLES = 2000000;
        int cycles = 0;
        
        fprintf(stderr, "DEBUG: Starting neuron processing...\n");
        
        // Process until DONE signal or timeout
        while (!dut->done && cycles < MAX_CYCLES) {
            // Check if hardware is requesting crossbar computation
            if (dut->xbar_compute && !dut->xbar_valid) {
                if (xbar_callback) {
                    uint16_t neuron_idx = dut->xbar_input_idx;
                    int16_t result = xbar_callback(neuron_idx, input_pixels);
                    
                    // Send response
                    dut->xbar_result = result;
                    dut->xbar_valid = 1;
                    debug_xbar_requests++;
                    debug_xbar_responses++;
                    
                    // Clock to let hardware see the response
                    clock_tick();
                    cycles++;
                    
                    // Clear valid after one cycle
                    dut->xbar_valid = 0;
                } else {
                    fprintf(stderr, "ERROR: No callback!\n");
                    return -3;
                }
            } else {
                // Normal clock tick
                clock_tick();
                cycles++;
            }
            
            // Capture outputs when available
            if (dut->output_valid) {
                uint16_t idx = dut->output_idx;
                if (idx < 256) {
                    output_neurons[idx] = (int16_t)dut->output_value;
                    neuron_count++;
                    
                    // ADD THIS CRITICAL CHECK:
                    if (neuron_count >= 256) {
                        // Force completion to prevent overflow
                        fprintf(stderr, "DEBUG: Reached 256 neurons, stopping\n");
                        dut->done = 1;  // Signal completion to hardware
                        break;  // Exit the while loop
                    }
                    
                    if (neuron_count % 50 == 0) {
                        fprintf(stderr, "DEBUG: Got %d neurons\n", neuron_count);
                    }
                }
            }
        }
        
        fprintf(stderr, "DEBUG: Neuron processing done - got %d neurons\n", neuron_count);

        // Check if we got all neurons
        if (neuron_count != 256) {
            fprintf(stderr, "WARNING: Got %d/256 neurons\n", neuron_count);
            fprintf(stderr, "  xbar_requests=%d, xbar_responses=%d\n",
                    debug_xbar_requests, debug_xbar_responses);
            fprintf(stderr, "  Final state: busy=%d, done=%d, xbar_compute=%d\n",
                    dut->busy, dut->done, dut->xbar_compute);
            
            // Try a few more cycles to see if we can complete
            for (int i = 0; i < 10 && !dut->done; i++) {
                clock_tick();
            }
        }

        // Wait for DONE signal
        int wait_done = 0;
        while (!dut->done && wait_done < 20) {
            clock_tick();
            wait_done++;
        }
        
        if (!dut->done) {
            fprintf(stderr, "ERROR: Hardware did not finish (done not asserted)\n");
        }

        return neuron_count;
    }

    uint64_t get_cycle_count() const {
        return sim_time / 2;
    }
};

// =====================================================
// C API
// =====================================================
extern "C" {

ReRAMHardware* hardware_create() {
    return new ReRAMHardware();
}

void hardware_destroy(ReRAMHardware* hw) {
    delete hw;
}

void hardware_set_callback(ReRAMHardware* hw,
                           int16_t (*callback)(uint16_t, const uint8_t*)) {
    hw->set_crossbar_callback(callback);
}

int hardware_run_inference(ReRAMHardware* hw,
                           const uint8_t* input_pixels,
                           int16_t* output_neurons) {
    return hw->run_inference(input_pixels, output_neurons);
}

uint64_t hardware_get_cycles(ReRAMHardware* hw) {
    return hw->get_cycle_count();
}

}
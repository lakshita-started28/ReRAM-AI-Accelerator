`timescale 1ns / 1ps

//==============================================================================
// Testbench for ReRAM Controller - FIXED VERSION
// Tests: Load input → DAC → Crossbar (mock) → ADC → Output
//==============================================================================

module tb_reram_controller;

    //--------------------------------------------------------------------------
    // Parameters
    //--------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;  // 10ns = 100MHz
    parameter INPUT_SIZE = 784;
    parameter OUTPUT_SIZE = 256;
    
    //--------------------------------------------------------------------------
    // DUT Signals
    //--------------------------------------------------------------------------
    reg                 clk;
    reg                 rst_n;
    reg                 start;
    wire                done;
    wire                busy;
    
    reg  [7:0]          input_data;
    reg                 input_valid;
    wire                input_ready;
    
    wire [9:0]          dac_out;
    wire                dac_valid;
    
    wire                xbar_enable;
    wire [9:0]          xbar_addr;
    reg  [11:0]         xbar_data;
    reg                 xbar_valid;
    
    wire [11:0]         output_data;
    wire [7:0]          output_addr;
    wire                output_valid;
    
    //--------------------------------------------------------------------------
    // Test Variables
    //--------------------------------------------------------------------------
    integer i;
    integer input_count;
    integer output_count;
    integer errors;
    
    // Test input data (simplified MNIST-like pattern)
    reg [7:0] test_input [0:INPUT_SIZE-1];
    reg [11:0] expected_output [0:OUTPUT_SIZE-1];
    
    //--------------------------------------------------------------------------
    // Clock Generation
    //--------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //--------------------------------------------------------------------------
    // DUT Instantiation
    //--------------------------------------------------------------------------
    reram_controller #(
        .INPUT_WIDTH(8),
        .OUTPUT_WIDTH(12),
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .input_data(input_data),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .dac_out(dac_out),
        .dac_valid(dac_valid),
        .xbar_enable(xbar_enable),
        .xbar_addr(xbar_addr),
        .xbar_data(xbar_data),
        .xbar_valid(xbar_valid),
        .output_data(output_data),
        .output_addr(output_addr),
        .output_valid(output_valid)
    );
    
    //--------------------------------------------------------------------------
    // Mock Crossbar Response (Simulates ReRAM computation)
    //--------------------------------------------------------------------------
    reg [3:0] xbar_delay_counter;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            xbar_valid <= 1'b0;
            xbar_data <= 12'b0;
            xbar_delay_counter <= 4'b0;
        end else begin
            xbar_valid <= 1'b0;  // Default
            
            if (xbar_enable) begin
                if (xbar_delay_counter < 4'd10) begin
                    xbar_delay_counter <= xbar_delay_counter + 1'b1;
                end else begin
                    // Mock computation: output = input * weight_factor
                    xbar_data <= (dac_out * 4) % 4096;
                    xbar_valid <= 1'b1;
                    xbar_delay_counter <= 4'b0;
                end
            end else begin
                xbar_delay_counter <= 4'b0;
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // Initialize Test Data
    //--------------------------------------------------------------------------
    initial begin
        // Generate test pattern (simulating MNIST digit "1")
        for (i = 0; i < INPUT_SIZE; i = i + 1) begin
            if (i >= 300 && i < 500) begin
                test_input[i] = 8'd255;  // Bright pixels in center
            end else begin
                test_input[i] = 8'd0;    // Dark pixels elsewhere
            end
        end
        
        // Expected outputs (mock - for verification)
        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
            expected_output[i] = 12'd2048;  // Placeholder
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Monitor (runs in parallel)
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (output_valid) begin
            output_count = output_count + 1;
            if (output_count % 50 == 0) begin
                $display("  Output %0d: addr=%0d, data=%0d", 
                       output_count, output_addr, output_data);
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // Test Stimulus
    //--------------------------------------------------------------------------
    initial begin
        // Initialize signals
        rst_n = 0;
        start = 0;
        input_data = 8'b0;
        input_valid = 0;
        input_count = 0;
        output_count = 0;
        errors = 0;
        
        // Dump waveforms
        $dumpfile("sim/reram_controller.vcd");
        $dumpvars(0, tb_reram_controller);
        
        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        
        $display("========================================");
        $display("ReRAM Controller Testbench");
        $display("========================================");
        $display("Time: %0t - Starting test...", $time);
        
        // Start inference
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        // Wait for controller to be ready
        while (busy == 0) begin
            @(posedge clk);
        end
        $display("Time: %0t - Controller active", $time);
        
        // Load input data
        $display("Time: %0t - Loading %0d inputs...", $time, INPUT_SIZE);
        for (i = 0; i < INPUT_SIZE; i = i + 1) begin
            // Wait for ready signal
            while (input_ready == 0) begin
                @(posedge clk);
            end
            
            @(posedge clk);
            input_data = test_input[i];
            input_valid = 1;
            @(posedge clk);
            input_valid = 0;
            
            if (i % 100 == 0) begin
                $display("  Loaded %0d/%0d inputs", i, INPUT_SIZE);
            end
        end
        
        $display("Time: %0t - All inputs loaded", $time);
        
        // Wait for completion
        while (done == 0) begin
            @(posedge clk);
        end
        
        $display("Time: %0t - Inference complete!", $time);
        $display("  Total outputs: %0d", output_count);
        
        // Check results
        #(CLK_PERIOD * 10);
        
        if (output_count == OUTPUT_SIZE) begin
            $display("");
            $display("+ TEST PASSED: All %0d outputs generated", OUTPUT_SIZE);
        end else begin
            $display("");
            $display("X TEST FAILED: Expected %0d outputs, got %0d", 
                   OUTPUT_SIZE, output_count);
            errors = errors + 1;
        end
        
        $display("========================================");
        $display("Test Summary:");
        $display("  Inputs loaded:  %0d", INPUT_SIZE);
        $display("  Outputs generated: %0d", output_count);
        $display("  Errors: %0d", errors);
        $display("========================================");
        
        if (errors == 0) begin
            $display("+ ALL TESTS PASSED");
        end else begin
            $display("X TESTS FAILED (%0d errors)", errors);
        end
        
        $finish;
    end
    
    //--------------------------------------------------------------------------
    // Timeout Watchdog
    //--------------------------------------------------------------------------
    initial begin
        #(CLK_PERIOD * 1000000);  // 10ms timeout
        $display("");
        $display("X ERROR: Test timeout!");
        $finish;
    end

endmodule
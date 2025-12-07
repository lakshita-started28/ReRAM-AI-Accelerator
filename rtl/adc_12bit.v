`timescale 1ns / 1ps

//==============================================================================
// 12-bit ADC Behavioral Model (Successive Approximation)
// Function: Converts analog current (from crossbar) to digital value
// Range: ±2.5V input → 0-4095 digital output (signed)
//==============================================================================

module adc_12bit #(
    parameter ADC_BITS = 12,
    parameter V_REF_MV = 2500,     // Reference voltage ±2.5V
    parameter CONV_CYCLES = 12     // Conversion time in clock cycles
)(
    input  wire                    clk,
    input  wire                    rst_n,
    
    // Analog input (represented as signed fixed-point in mV)
    input  wire signed [15:0]      analog_in_mv,  // Signed voltage in mV
    input  wire                    start_conv,    // Start conversion
    
    // Digital output
    output reg  signed [ADC_BITS-1:0] digital_out,
    output reg                        valid_out,
    output reg                        busy
);

    //--------------------------------------------------------------------------
    // ADC Parameters
    //--------------------------------------------------------------------------
    localparam MAX_CODE = (1 << ADC_BITS) - 1;  // 4095 for 12-bit
    localparam HALF_RANGE = (1 << (ADC_BITS - 1));  // 2048 for signed
    
    // LSB size: (2 * V_REF) / 2^n
    // For 12-bit, ±2.5V: LSB = 5000mV / 4096 ≈ 1.22 mV
    
    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    reg [3:0] conv_counter;  // Conversion cycle counter
    reg [2:0] adc_state;
    reg signed [15:0] voltage_latched;
    
    // ADC States
    localparam IDLE        = 3'b000;
    localparam SAMPLE      = 3'b001;
    localparam HOLD        = 3'b010;
    localparam CONVERT     = 3'b011;
    localparam QUANTIZE    = 3'b100;
    localparam OUTPUT      = 3'b101;
    
    //--------------------------------------------------------------------------
    // ADC Conversion FSM
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            adc_state      <= IDLE;
            digital_out    <= {ADC_BITS{1'b0}};
            valid_out      <= 1'b0;
            busy           <= 1'b0;
            conv_counter   <= 4'b0;
            voltage_latched <= 16'b0;
        end else begin
            case (adc_state)
                IDLE: begin
                    valid_out    <= 1'b0;
                    busy         <= 1'b0;
                    conv_counter <= 4'b0;
                    
                    if (start_conv) begin
                        busy      <= 1'b1;
                        adc_state <= SAMPLE;
                    end
                end
                
                SAMPLE: begin
                    // Sample the input voltage
                    voltage_latched <= analog_in_mv;
                    adc_state       <= HOLD;
                end
                
                HOLD: begin
                    // Hold the sampled value
                    adc_state <= CONVERT;
                end
                
                CONVERT: begin
                    // Simulate SAR conversion cycles
                    if (conv_counter < CONV_CYCLES) begin
                        conv_counter <= conv_counter + 1'b1;
                    end else begin
                        adc_state    <= QUANTIZE;
                        conv_counter <= 4'b0;
                    end
                end
                
                QUANTIZE: begin
                    // Quantize voltage to digital code
                    // Formula: code = (V_in / V_ref) * MAX_CODE
                    
                    // Clamp input to ±V_REF
                    if (voltage_latched > V_REF_MV) begin
                        digital_out <= HALF_RANGE - 1;  // Max positive
                    end else if (voltage_latched < -V_REF_MV) begin
                        digital_out <= -HALF_RANGE;     // Max negative
                    end else begin
                        // Linear conversion with proper rounding
                        // Convert to 12-bit signed: [-2048, 2047]
                        digital_out <= (voltage_latched * HALF_RANGE) / V_REF_MV;
                    end
                    
                    adc_state <= OUTPUT;
                end
                
                OUTPUT: begin
                    valid_out <= 1'b1;
                    busy      <= 1'b0;
                    adc_state <= IDLE;
                end
                
                default: begin
                    adc_state <= IDLE;
                end
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Non-Ideality Modeling (Optional)
    //--------------------------------------------------------------------------
    // Could add:
    // 1. Quantization noise
    // 2. Aperture jitter
    // 3. INL/DNL errors
    // 4. Comparator offset
    
    //--------------------------------------------------------------------------
    // Debug Signals
    //--------------------------------------------------------------------------
    // synthesis translate_off
    real voltage_real, code_real;
    
    always @(*) begin
        voltage_real = voltage_latched / 1000.0;  // Convert to volts
        code_real    = $itor(digital_out);
    end
    // synthesis translate_on

endmodule
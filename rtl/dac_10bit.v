`timescale 1ns / 1ps

//==============================================================================
// 10-bit DAC Behavioral Model
// Function: Converts digital input to analog voltage
// Range: 0-1023 digital → 0.0V-0.3V analog (for ReRAM read voltage)
//==============================================================================

module dac_10bit #(
    parameter DAC_BITS = 10,
    parameter V_REF_MV = 300  // Reference voltage in millivolts (0.3V)
)(
    input  wire                    clk,
    input  wire                    rst_n,
    
    // Digital input
    input  wire [DAC_BITS-1:0]     digital_in,
    input  wire                    valid_in,
    
    // Analog output (represented as fixed-point in mV)
    output reg  [15:0]             analog_out_mv,  // Voltage in millivolts
    output reg                     valid_out
);

    //--------------------------------------------------------------------------
    // DAC Conversion Parameters
    //--------------------------------------------------------------------------
    localparam MAX_CODE = (1 << DAC_BITS) - 1;  // 1023 for 10-bit
    
    // LSB size in mV: V_REF / 2^n
    // For 10-bit, 300mV: LSB = 300/1024 ≈ 0.293 mV
    localparam LSB_SIZE_MV = V_REF_MV / (1 << DAC_BITS);
    
    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    reg [1:0] conversion_state;
    reg [DAC_BITS-1:0] digital_code;
    
    // Conversion states
    localparam IDLE      = 2'b00;
    localparam CONVERT   = 2'b01;
    localparam SETTLING  = 2'b10;
    localparam OUTPUT    = 2'b11;
    
    //--------------------------------------------------------------------------
    // DAC Conversion FSM
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conversion_state <= IDLE;
            analog_out_mv    <= 16'b0;
            valid_out        <= 1'b0;
            digital_code     <= {DAC_BITS{1'b0}};
        end else begin
            case (conversion_state)
                IDLE: begin
                    valid_out <= 1'b0;
                    
                    if (valid_in) begin
                        digital_code     <= digital_in;
                        conversion_state <= CONVERT;
                    end
                end
                
                CONVERT: begin
                    // Ideal DAC transfer function:
                    // V_out = (digital_code / MAX_CODE) * V_REF
                    // In mV: V_out = (digital_code * V_REF_MV) / MAX_CODE
                    
                    analog_out_mv <= (digital_code * V_REF_MV) / MAX_CODE;
                    conversion_state <= SETTLING;
                end
                
                SETTLING: begin
                    // DAC settling time (1 cycle)
                    conversion_state <= OUTPUT;
                end
                
                OUTPUT: begin
                    valid_out        <= 1'b1;
                    conversion_state <= IDLE;
                end
                
                default: begin
                    conversion_state <= IDLE;
                end
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Non-Ideality Modeling (Optional - for realism)
    //--------------------------------------------------------------------------
    // In production, you might add:
    // 1. INL (Integral Non-Linearity)
    // 2. DNL (Differential Non-Linearity)
    // 3. Offset error
    // 4. Gain error
    // 5. Noise
    
    // For now, we use ideal conversion
    
    //--------------------------------------------------------------------------
    // Debug/Monitoring Signals
    //--------------------------------------------------------------------------
    // synthesis translate_off
    real voltage_real;
    
    always @(analog_out_mv) begin
        voltage_real = analog_out_mv / 1000.0;  // Convert to volts
    end
    // synthesis translate_on

endmodule
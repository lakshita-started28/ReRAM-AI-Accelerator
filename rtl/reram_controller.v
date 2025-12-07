`timescale 1ns / 1ps

//==============================================================================
// ReRAM Accelerator Controller - FIXED FSM
// Function: Controls the dataflow for ReRAM-based inference
//==============================================================================

module reram_controller #(
    parameter INPUT_WIDTH = 8,
    parameter OUTPUT_WIDTH = 12,
    parameter INPUT_SIZE = 784,
    parameter OUTPUT_SIZE = 256
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control signals
    input  wire                     start,
    output reg                      done,
    output reg                      busy,
    
    // Input interface
    input  wire [INPUT_WIDTH-1:0]   input_data,
    input  wire                     input_valid,
    output reg                      input_ready,
    
    // DAC interface
    output reg  [9:0]               dac_out,
    output reg                      dac_valid,
    
    // Crossbar interface
    output reg                      xbar_enable,
    output reg  [9:0]               xbar_addr,
    input  wire [OUTPUT_WIDTH-1:0]  xbar_data,
    input  wire                     xbar_valid,
    
    // Output interface
    output reg  [OUTPUT_WIDTH-1:0]  output_data,
    output reg  [7:0]               output_addr,
    output reg                      output_valid
);

    //--------------------------------------------------------------------------
    // FSM States
    //--------------------------------------------------------------------------
    localparam IDLE         = 3'b000;
    localparam LOAD_INPUT   = 3'b001;
    localparam COMPUTE_WAIT = 3'b010;
    localparam COMPUTE_DONE = 3'b011;
    localparam OUTPUT_WRITE = 3'b100;
    localparam DONE_STATE   = 3'b101;
    
    reg [2:0] state, next_state;
    
    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    reg [9:0]  input_counter;
    reg [8:0]  output_counter;  // Changed to 9 bits to hold 256
    reg [15:0] wait_counter;
    
    // Input buffer
    reg [INPUT_WIDTH-1:0] input_buffer [0:INPUT_SIZE-1];
    
    // Output buffer
    reg [OUTPUT_WIDTH-1:0] output_buffer [0:OUTPUT_SIZE-1];
    
    // Computation state
    reg computing;
    reg [9:0] current_neuron;
    
    //--------------------------------------------------------------------------
    // State Machine - Sequential
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    //--------------------------------------------------------------------------
    // State Machine - Combinational (Next State Logic)
    //--------------------------------------------------------------------------
    always @(*) begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) begin
                    next_state = LOAD_INPUT;
                end
            end
            
            LOAD_INPUT: begin
                if (input_counter == INPUT_SIZE && input_valid) begin
                    next_state = COMPUTE_WAIT;
                end
            end
            
            COMPUTE_WAIT: begin
                if (xbar_valid) begin
                    next_state = COMPUTE_DONE;
                end
            end
            
            COMPUTE_DONE: begin
                next_state = OUTPUT_WRITE;
            end
            
            OUTPUT_WRITE: begin
                if (current_neuron == OUTPUT_SIZE - 1) begin
                    next_state = DONE_STATE;
                end else begin
                    next_state = COMPUTE_WAIT;
                end
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // State Machine - Output Logic
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done            <= 1'b0;
            busy            <= 1'b0;
            input_ready     <= 1'b0;
            dac_out         <= 10'b0;
            dac_valid       <= 1'b0;
            xbar_enable     <= 1'b0;
            xbar_addr       <= 10'b0;
            output_data     <= {OUTPUT_WIDTH{1'b0}};
            output_addr     <= 8'b0;
            output_valid    <= 1'b0;
            input_counter   <= 10'b0;
            output_counter  <= 9'b0;
            wait_counter    <= 16'b0;
            computing       <= 1'b0;
            current_neuron  <= 10'b0;
        end else begin
            // Default outputs
            input_ready  <= 1'b0;
            dac_valid    <= 1'b0;
            xbar_enable  <= 1'b0;
            output_valid <= 1'b0;
            done         <= 1'b0;
            
            case (state)
                IDLE: begin
                    busy           <= 1'b0;
                    input_counter  <= 10'b0;
                    output_counter <= 9'b0;
                    wait_counter   <= 16'b0;
                    computing      <= 1'b0;
                    current_neuron <= 10'b0;
                    
                    if (start) begin
                        busy <= 1'b1;
                    end
                end
                
                LOAD_INPUT: begin
                    busy        <= 1'b1;
                    input_ready <= 1'b1;
                    
                    if (input_valid) begin
                        input_buffer[input_counter] <= input_data;
                        
                        if (input_counter < INPUT_SIZE) begin
                            input_counter <= input_counter + 1'b1;
                        end
                    end
                end
                
                COMPUTE_WAIT: begin
                    busy        <= 1'b1;
                    xbar_enable <= 1'b1;
                    xbar_addr   <= current_neuron;
                    
                    // Send DAC output (use first input for now - simplified)
                    dac_out   <= {input_buffer[0], 2'b00};
                    dac_valid <= 1'b1;
                end
                
                COMPUTE_DONE: begin
                    busy <= 1'b1;
                    // Store result
                    output_buffer[current_neuron] <= xbar_data;
                end
                
                OUTPUT_WRITE: begin
                    busy         <= 1'b1;
                    output_data  <= output_buffer[current_neuron];
                    output_addr  <= current_neuron[7:0];
                    output_valid <= 1'b1;
                    
                    if (current_neuron < OUTPUT_SIZE - 1) begin
                        current_neuron <= current_neuron + 1'b1;
                    end
                end
                
                DONE_STATE: begin
                    done            <= 1'b1;
                    busy            <= 1'b0;
                    input_counter   <= 10'b0;
                    output_counter  <= 9'b0;
                    current_neuron  <= 10'b0;
                end
                
                default: begin
                    busy <= 1'b0;
                end
            endcase
        end
    end

endmodule
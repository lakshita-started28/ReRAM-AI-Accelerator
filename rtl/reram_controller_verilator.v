`timescale 1ns / 1ps

module reram_controller_verilator #(
    parameter INPUT_SIZE = 784,
    parameter OUTPUT_SIZE = 256
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    output reg busy,
    output reg done,
    input wire [7:0] input_pixel,
    input wire input_valid,
    output reg input_ready,
    output reg xbar_compute,
    output reg [15:0] xbar_input_idx,
    input wire signed [15:0] xbar_result,
    input wire xbar_valid,
    output reg output_valid,
    output reg [7:0] output_idx,
    output reg signed [15:0] output_value
);

    localparam IDLE = 3'd0, LOAD = 3'd1, COMPUTE = 3'd2, WAIT = 3'd3, OUTPUT = 3'd4, DONE = 3'd5;
    
    reg [2:0] state;
    reg [9:0] pixel_count; 
    reg [8:0] neuron_count;
    reg [7:0] pixels [0:783];
    reg signed [15:0] results [0:255];
    
    // FSM and all outputs updated on clock edge
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            busy <= 0;
            done <= 0;
            input_ready <= 0;
            xbar_compute <= 0;
            output_valid <= 0;
            pixel_count <= 0;
            neuron_count <= 0;
            xbar_input_idx <= 0;
            output_idx <= 0;
            output_value <= 0;
        end else begin
            // Default outputs for next cycle
            input_ready <= 0;
            xbar_compute <= 0;
            output_valid <= 0;
            done <= 0;
            
            case (state)
                IDLE: begin
                    busy <= 0;
                    pixel_count <= 0;
                    neuron_count <= 0;
                    if (start) begin
                        state <= LOAD;
                        busy <= 1;
                    end
                end
                
                LOAD: begin
                    input_ready <= 1;
                    
                    if (input_valid) begin
                        pixels[pixel_count] <= input_pixel;
                        if (pixel_count == (INPUT_SIZE - 1)) begin
                            state <= COMPUTE;
                        end else begin
                            pixel_count <= pixel_count + 1;
                        end
                    end
                end
                
                COMPUTE: begin
                    xbar_compute <= 1;
                    xbar_input_idx <= {8'd0, neuron_count[7:0]};
                    state <= WAIT;
                end
                
                WAIT: begin
                    if (xbar_valid) begin
                        results[neuron_count[7:0]] <= xbar_result;
                        state <= OUTPUT;
                    end
                end
                
                OUTPUT: begin
                    output_valid <= 1;
                    output_idx <= neuron_count[7:0];
                    output_value <= results[neuron_count[7:0]];
                    
                    if (neuron_count == (OUTPUT_SIZE - 1)) begin
                        state <= DONE;
                    end else begin
                        neuron_count <= neuron_count + 1;
                        state <= COMPUTE;
                    end
                end
                
                DONE: begin
                    done <= 1;
                    busy <= 0;
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
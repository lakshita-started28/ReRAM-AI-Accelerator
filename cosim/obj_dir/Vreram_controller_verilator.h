// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VRERAM_CONTROLLER_VERILATOR_H_
#define _VRERAM_CONTROLLER_VERILATOR_H_  // guard

#include "verilated.h"

//==========

class Vreram_controller_verilator__Syms;

//----------

VL_MODULE(Vreram_controller_verilator) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(start,0,0);
    VL_OUT8(busy,0,0);
    VL_OUT8(done,0,0);
    VL_IN8(input_pixel,7,0);
    VL_IN8(input_valid,0,0);
    VL_OUT8(input_ready,0,0);
    VL_OUT8(xbar_compute,0,0);
    VL_IN8(xbar_valid,0,0);
    VL_OUT8(output_valid,0,0);
    VL_OUT8(output_idx,7,0);
    VL_OUT16(xbar_input_idx,15,0);
    VL_IN16(xbar_result,15,0);
    VL_OUT16(output_value,15,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    CData/*2:0*/ reram_controller_verilator__DOT__state;
    SData/*9:0*/ reram_controller_verilator__DOT__pixel_count;
    SData/*8:0*/ reram_controller_verilator__DOT__neuron_count;
    CData/*7:0*/ reram_controller_verilator__DOT__pixels[784];
    SData/*15:0*/ reram_controller_verilator__DOT__results[256];
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    CData/*7:0*/ reram_controller_verilator__DOT____Vlvbound1;
    CData/*0:0*/ __Vclklast__TOP__clk;
    CData/*0:0*/ __Vclklast__TOP__rst_n;
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vreram_controller_verilator__Syms* __VlSymsp;  // Symbol table
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vreram_controller_verilator);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    Vreram_controller_verilator(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vreram_controller_verilator();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval() { eval_step(); }
    /// Evaluate when calling multiple units/models per time step.
    void eval_step();
    /// Evaluate at end of a timestep for tracing, when using eval_step().
    /// Application must call after all eval() and before time changes.
    void eval_end_step() {}
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vreram_controller_verilator__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vreram_controller_verilator__Syms* symsp, bool first);
  private:
    static QData _change_request(Vreram_controller_verilator__Syms* __restrict vlSymsp);
    static QData _change_request_1(Vreram_controller_verilator__Syms* __restrict vlSymsp);
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void _eval(Vreram_controller_verilator__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif  // VL_DEBUG
  public:
    static void _eval_initial(Vreram_controller_verilator__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _eval_settle(Vreram_controller_verilator__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _sequent__TOP__1(Vreram_controller_verilator__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

//----------


#endif  // guard

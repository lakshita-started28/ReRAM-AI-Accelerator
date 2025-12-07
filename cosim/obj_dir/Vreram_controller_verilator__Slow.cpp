// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vreram_controller_verilator.h for the primary calling header

#include "Vreram_controller_verilator.h"
#include "Vreram_controller_verilator__Syms.h"

//==========

VL_CTOR_IMP(Vreram_controller_verilator) {
    Vreram_controller_verilator__Syms* __restrict vlSymsp = __VlSymsp = new Vreram_controller_verilator__Syms(this, name());
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vreram_controller_verilator::__Vconfigure(Vreram_controller_verilator__Syms* vlSymsp, bool first) {
    if (false && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
    if (false && this->__VlSymsp) {}  // Prevent unused
    Verilated::timeunit(-9);
    Verilated::timeprecision(-12);
}

Vreram_controller_verilator::~Vreram_controller_verilator() {
    VL_DO_CLEAR(delete __VlSymsp, __VlSymsp = NULL);
}

void Vreram_controller_verilator::_eval_initial(Vreram_controller_verilator__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_eval_initial\n"); );
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__rst_n = vlTOPp->rst_n;
}

void Vreram_controller_verilator::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::final\n"); );
    // Variables
    Vreram_controller_verilator__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vreram_controller_verilator::_eval_settle(Vreram_controller_verilator__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_eval_settle\n"); );
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vreram_controller_verilator::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_ctor_var_reset\n"); );
    // Body
    clk = 0;
    rst_n = 0;
    start = 0;
    busy = 0;
    done = 0;
    input_pixel = 0;
    input_valid = 0;
    input_ready = 0;
    xbar_compute = 0;
    xbar_input_idx = 0;
    xbar_result = 0;
    xbar_valid = 0;
    output_valid = 0;
    output_idx = 0;
    output_value = 0;
    reram_controller_verilator__DOT__state = 0;
    reram_controller_verilator__DOT__pixel_count = 0;
    reram_controller_verilator__DOT__neuron_count = 0;
    { int __Vi0=0; for (; __Vi0<784; ++__Vi0) {
            reram_controller_verilator__DOT__pixels[__Vi0] = 0;
    }}
    { int __Vi0=0; for (; __Vi0<256; ++__Vi0) {
            reram_controller_verilator__DOT__results[__Vi0] = 0;
    }}
    reram_controller_verilator__DOT____Vlvbound1 = 0;
}

// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vreram_controller_verilator.h for the primary calling header

#include "Vreram_controller_verilator.h"
#include "Vreram_controller_verilator__Syms.h"

//==========

void Vreram_controller_verilator::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vreram_controller_verilator::eval\n"); );
    Vreram_controller_verilator__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
#ifdef VL_DEBUG
    // Debug assertions
    _eval_debug_assertions();
#endif  // VL_DEBUG
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Clock loop\n"););
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("../rtl/reram_controller_verilator.v", 3, "",
                "Verilated model didn't converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

void Vreram_controller_verilator::_eval_initial_loop(Vreram_controller_verilator__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    _eval_initial(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        _eval_settle(vlSymsp);
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("../rtl/reram_controller_verilator.v", 3, "",
                "Verilated model didn't DC converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

VL_INLINE_OPT void Vreram_controller_verilator::_sequent__TOP__1(Vreram_controller_verilator__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_sequent__TOP__1\n"); );
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    CData/*2:0*/ __Vdly__reram_controller_verilator__DOT__state;
    CData/*7:0*/ __Vdlyvdim0__reram_controller_verilator__DOT__results__v0;
    CData/*0:0*/ __Vdlyvset__reram_controller_verilator__DOT__results__v0;
    CData/*7:0*/ __Vdlyvval__reram_controller_verilator__DOT__pixels__v0;
    CData/*0:0*/ __Vdlyvset__reram_controller_verilator__DOT__pixels__v0;
    SData/*8:0*/ __Vdly__reram_controller_verilator__DOT__neuron_count;
    SData/*15:0*/ __Vdlyvval__reram_controller_verilator__DOT__results__v0;
    SData/*9:0*/ __Vdlyvdim0__reram_controller_verilator__DOT__pixels__v0;
    // Body
    __Vdly__reram_controller_verilator__DOT__neuron_count 
        = vlTOPp->reram_controller_verilator__DOT__neuron_count;
    __Vdly__reram_controller_verilator__DOT__state 
        = vlTOPp->reram_controller_verilator__DOT__state;
    __Vdlyvset__reram_controller_verilator__DOT__pixels__v0 = 0U;
    __Vdlyvset__reram_controller_verilator__DOT__results__v0 = 0U;
    if (vlTOPp->rst_n) {
        vlTOPp->input_ready = 0U;
        vlTOPp->xbar_compute = 0U;
        vlTOPp->output_valid = 0U;
        vlTOPp->done = 0U;
        if ((4U & (IData)(vlTOPp->reram_controller_verilator__DOT__state))) {
            if ((2U & (IData)(vlTOPp->reram_controller_verilator__DOT__state))) {
                __Vdly__reram_controller_verilator__DOT__state = 0U;
            } else {
                if ((1U & (IData)(vlTOPp->reram_controller_verilator__DOT__state))) {
                    vlTOPp->done = 1U;
                    vlTOPp->busy = 0U;
                    __Vdly__reram_controller_verilator__DOT__state = 0U;
                } else {
                    vlTOPp->output_valid = 1U;
                    vlTOPp->output_idx = (0xffU & (IData)(vlTOPp->reram_controller_verilator__DOT__neuron_count));
                    vlTOPp->output_value = vlTOPp->reram_controller_verilator__DOT__results
                        [(0xffU & (IData)(vlTOPp->reram_controller_verilator__DOT__neuron_count))];
                    if ((0xffU == (IData)(vlTOPp->reram_controller_verilator__DOT__neuron_count))) {
                        __Vdly__reram_controller_verilator__DOT__state = 5U;
                    } else {
                        __Vdly__reram_controller_verilator__DOT__neuron_count 
                            = (0x1ffU & ((IData)(1U) 
                                         + (IData)(vlTOPp->reram_controller_verilator__DOT__neuron_count)));
                        __Vdly__reram_controller_verilator__DOT__state = 2U;
                    }
                }
            }
        } else {
            if ((2U & (IData)(vlTOPp->reram_controller_verilator__DOT__state))) {
                if ((1U & (IData)(vlTOPp->reram_controller_verilator__DOT__state))) {
                    if (vlTOPp->xbar_valid) {
                        __Vdlyvval__reram_controller_verilator__DOT__results__v0 
                            = vlTOPp->xbar_result;
                        __Vdlyvset__reram_controller_verilator__DOT__results__v0 = 1U;
                        __Vdlyvdim0__reram_controller_verilator__DOT__results__v0 
                            = (0xffU & (IData)(vlTOPp->reram_controller_verilator__DOT__neuron_count));
                        __Vdly__reram_controller_verilator__DOT__state = 4U;
                    }
                } else {
                    vlTOPp->xbar_compute = 1U;
                    vlTOPp->xbar_input_idx = (0xffU 
                                              & (IData)(vlTOPp->reram_controller_verilator__DOT__neuron_count));
                    __Vdly__reram_controller_verilator__DOT__state = 3U;
                }
            } else {
                if ((1U & (IData)(vlTOPp->reram_controller_verilator__DOT__state))) {
                    vlTOPp->input_ready = 1U;
                    if (vlTOPp->input_valid) {
                        vlTOPp->reram_controller_verilator__DOT____Vlvbound1 
                            = vlTOPp->input_pixel;
                        if ((0x30fU >= (IData)(vlTOPp->reram_controller_verilator__DOT__pixel_count))) {
                            __Vdlyvval__reram_controller_verilator__DOT__pixels__v0 
                                = vlTOPp->reram_controller_verilator__DOT____Vlvbound1;
                            __Vdlyvset__reram_controller_verilator__DOT__pixels__v0 = 1U;
                            __Vdlyvdim0__reram_controller_verilator__DOT__pixels__v0 
                                = vlTOPp->reram_controller_verilator__DOT__pixel_count;
                        }
                        if ((0x30fU == (IData)(vlTOPp->reram_controller_verilator__DOT__pixel_count))) {
                            __Vdly__reram_controller_verilator__DOT__state = 2U;
                        } else {
                            vlTOPp->reram_controller_verilator__DOT__pixel_count 
                                = (0x3ffU & ((IData)(1U) 
                                             + (IData)(vlTOPp->reram_controller_verilator__DOT__pixel_count)));
                        }
                    }
                } else {
                    __Vdly__reram_controller_verilator__DOT__neuron_count = 0U;
                    vlTOPp->busy = 0U;
                    vlTOPp->reram_controller_verilator__DOT__pixel_count = 0U;
                    if (vlTOPp->start) {
                        __Vdly__reram_controller_verilator__DOT__state = 1U;
                        vlTOPp->busy = 1U;
                    }
                }
            }
        }
    } else {
        __Vdly__reram_controller_verilator__DOT__neuron_count = 0U;
        __Vdly__reram_controller_verilator__DOT__state = 0U;
        vlTOPp->busy = 0U;
        vlTOPp->done = 0U;
        vlTOPp->input_ready = 0U;
        vlTOPp->xbar_compute = 0U;
        vlTOPp->output_valid = 0U;
        vlTOPp->reram_controller_verilator__DOT__pixel_count = 0U;
        vlTOPp->xbar_input_idx = 0U;
        vlTOPp->output_idx = 0U;
        vlTOPp->output_value = 0U;
    }
    vlTOPp->reram_controller_verilator__DOT__state 
        = __Vdly__reram_controller_verilator__DOT__state;
    vlTOPp->reram_controller_verilator__DOT__neuron_count 
        = __Vdly__reram_controller_verilator__DOT__neuron_count;
    if (__Vdlyvset__reram_controller_verilator__DOT__results__v0) {
        vlTOPp->reram_controller_verilator__DOT__results[__Vdlyvdim0__reram_controller_verilator__DOT__results__v0] 
            = __Vdlyvval__reram_controller_verilator__DOT__results__v0;
    }
    if (__Vdlyvset__reram_controller_verilator__DOT__pixels__v0) {
        vlTOPp->reram_controller_verilator__DOT__pixels[__Vdlyvdim0__reram_controller_verilator__DOT__pixels__v0] 
            = __Vdlyvval__reram_controller_verilator__DOT__pixels__v0;
    }
}

void Vreram_controller_verilator::_eval(Vreram_controller_verilator__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_eval\n"); );
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    if ((((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk))) 
         | ((~ (IData)(vlTOPp->rst_n)) & (IData)(vlTOPp->__Vclklast__TOP__rst_n)))) {
        vlTOPp->_sequent__TOP__1(vlSymsp);
    }
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__rst_n = vlTOPp->rst_n;
}

VL_INLINE_OPT QData Vreram_controller_verilator::_change_request(Vreram_controller_verilator__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_change_request\n"); );
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    return (vlTOPp->_change_request_1(vlSymsp));
}

VL_INLINE_OPT QData Vreram_controller_verilator::_change_request_1(Vreram_controller_verilator__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_change_request_1\n"); );
    Vreram_controller_verilator* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void Vreram_controller_verilator::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vreram_controller_verilator::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY((start & 0xfeU))) {
        Verilated::overWidthError("start");}
    if (VL_UNLIKELY((input_valid & 0xfeU))) {
        Verilated::overWidthError("input_valid");}
    if (VL_UNLIKELY((xbar_valid & 0xfeU))) {
        Verilated::overWidthError("xbar_valid");}
}
#endif  // VL_DEBUG

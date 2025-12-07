// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VRERAM_CONTROLLER_VERILATOR__SYMS_H_
#define _VRERAM_CONTROLLER_VERILATOR__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "Vreram_controller_verilator.h"

// SYMS CLASS
class Vreram_controller_verilator__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    Vreram_controller_verilator*   TOPp;
    
    // CREATORS
    Vreram_controller_verilator__Syms(Vreram_controller_verilator* topp, const char* namep);
    ~Vreram_controller_verilator__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard

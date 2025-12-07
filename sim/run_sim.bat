@echo off
echo =========================================
echo ReRAM Controller Simulation
echo =========================================

REM Clean previous simulation
if exist sim\*.vcd del /q sim\*.vcd
if exist sim\*.vvp del /q sim\*.vvp

REM Compile Verilog files with Icarus Verilog
echo Compiling Verilog files...
iverilog -o sim\reram_sim.vvp rtl\reram_controller.v rtl\dac_10bit.v rtl\adc_12bit.v testbench\tb_reram_controller.v

if %errorlevel% neq 0 (
    echo X Compilation failed!
    pause
    exit /b 1
)

echo + Compilation successful

REM Run simulation
echo Running simulation...
vvp sim\reram_sim.vvp

if %errorlevel% neq 0 (
    echo X Simulation failed!
    pause
    exit /b 1
)

echo.
echo + Simulation complete
echo Waveform saved to: sim\reram_controller.vcd
echo.
echo To view waveform:
echo   gtkwave sim\reram_controller.vcd
echo.
pause
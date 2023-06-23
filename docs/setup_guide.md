# Overview

This guide provides instructions on how to setup control software on the mantis microscope.

## Install Ti2 Control

Ti2 Control provides computer control of the Ti2 microscope body. The software can be obtained from our Nikon representative, [online](https://nisdk.recollective.com/microscopes), or on ESS at `/comp_micro/software/Microscope Hardware/Nikon Ti2`. We currently use version 1.2.0; version 2 should also work based on the discussion [here](https://github.com/micro-manager/mmCoreAndDevices/issues/44).

## Install SpinView

SpinView provides control of [FLIR](https://www.flir.com/) cameras on the microscope. Only version 2.3.0.77 works with the Micro-manager device adapter, as described [here](https://micro-manager.org/Spinnaker). A copy of the installer is available on ESS at `/comp_micro/software/microscope_hardware_installers/FLIR/SpinnakerSDK`

## Install CellDrive

CellDrive provides control of the [Meadowlark Optics](https://www.meadowlark.com/) liquid crystals (LCs) on the microscope. Install the latest version of the software which can be obtained from our Meadowlark representative. Copies of the installer are also available on ESS at TODO. We currently use software version 1.04. Ensure that both LCs can be controlled through CellDrive. The mantis acquisition controls the LCs via external voltage input - put both LCs in External Input mode.

## TODO: TriggerScope Firmware

## TODO: Vortran lasers

## TODO: Thorlabs Kinesis

## Install Micro-manager

The mantis acquisition currently works with Micro-manager nightly build 2023-03-12.

* Download [Micro-Manager nightly build 2023-03-12](https://download.micro-manager.org/nightly/2.0/Windows/MMSetup_64bit_2.0.1_20230312.exe) and install following the installer prompts in the `C:\Program Files\Micro-Manager-2.0_03_12_2023` directory.
  
  * Copy the `Ti2_Mic_Driver.dll` device adapter located at `C:\Program Files\Nikon\Ti2-SDK\bin` to the Micro-manager install directory, as described [here](https://micro-manager.org/NikonTi2).

* Install another copy of [Micro-Manager nightly build 2023-03-12](https://download.micro-manager.org/nightly/2.0/Windows/MMSetup_64bit_2.0.1_20230312.exe) in the `C:\Program Files\Micro-Manager-nightly` directory. This copy of Micro-manager will be used by the headless light-sheet acquisition engine.

## Test the hardware setup

* Launch Micro-manager using the `mantis-LS.cfg` and `mantis-LF.cfg` config files. You should not encounter any errors during startup.

* Run the mantis acquisition in demo mode with

```pwsh
mantis run-acquisition `
    --data-dirpath path/to/data/directory `
    --name test_acquisition `
    --mm-config-file path/to/MMConfig_Demo.cfg `
    --settings mantis/acquisition/settings/demo_acquisition_settings.yaml
```

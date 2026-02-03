# Overview

This guide provides instructions on how to setup control software on the mantis microscope.

## Install Ti2 Control

Ti2 Control provides computer control of the Ti2 microscope body. The software can be obtained from our Nikon representative, [online](https://nisdk.recollective.com/microscopes), or on ESS at `/comp_micro/software/Microscope Hardware/Nikon Ti2`. We currently use version 1.2.0; version 2 should also work based on the discussion [here](https://github.com/micro-manager/mmCoreAndDevices/issues/44).

## Install SpinView

SpinView provides control of [FLIR](https://www.flir.com/) cameras on the microscope. Only version 2.3.0.77 works with the Micro-manager device adapter, as described [here](https://micro-manager.org/Spinnaker). A copy of the installer is available on ESS at `/comp_micro/software/microscope_hardware_installers/FLIR/SpinnakerSDK`

## Install CellDrive

CellDrive provides control of the [Meadowlark Optics](https://www.meadowlark.com/) liquid crystals (LCs) on the microscope. Install the latest version of the software which can be obtained from our Meadowlark representative. Copies of the installer are also available on ESS at TODO. We currently use software version 1.04. Ensure that both LCs can be controlled through CellDrive. The mantis acquisition controls the LCs via external voltage input - put both LCs in External Input mode.

## Update TriggerScope Firmware

The [TriggerScope](https://advancedresearch-consulting.com/product/triggerscope-4/) is used to generate electrical signals which synchronize the acquisition on the mantis microscope. Firmware for the TriggerScope is available on the [micro-manager/TriggerScopeMM](https://github.com/micro-manager/TriggerScopeMM) GitHub repo. Different versions of the TriggerScope board are available - for example V3, V4, V4B. Be sure to follow the installation instruction and install the right firmware version for a given TriggerScope board. To track the version of the firmware that is currently installed, write the 7 digit Git hash of the last commit.

## Install Vortran Stradus

The [Vortran](https://www.vortranlaser.com/) lasers on the mantis microscope are controlled using the Stradus GUI application. Install the latest version of the software which can be obtained from our Vortran representative or on ESS at `software\StradusV4_0_0`. We currently use software version 4.0.0. The mantis acquisition engine uses [coPylot](https://github.com/czbiohub-sf/coPylot) to control the lasers during acquisition.

TODO: Update setup instructions to reflect requirements for coPylot control.

## Install Thorlabs Kinesis

Thorlabs Kinesis is used to control the [PIA13](https://www.thorlabs.com/thorproduct.cfm?partnumber=PIA13) stages for positioning objectives in the microscope remote refocus paths. Install the latest version of the software available from the manufacturer's [website](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285). We currently use version 1.14.30. The mantis acquisition engine uses [coPylot](https://github.com/czbiohub-sf/coPylot) to control the stages during acquisition.

TODO: Update the setup instructions to reflect requirements for coPylot control.

## Install Micro-manager

The Micro-manager nightly build tested and recommended for the mantis acquisition engine is declared in [`mantis/__init__.py`](../mantis/__init__.py) and can be obtained with:

```python
import mantis
print(mantis.__mm_version__)
```

* Download the recommended [Micro-Manager nightly build](https://download.micro-manager.org/nightly/2.0/Windows/) and install following the installer prompts in the `C:\Program Files\Micro-Manager-2.0_YYYY_MM_DD` directory.

  * Copy the `Ti2_Mic_Driver.dll` device adapter located at `C:\Program Files\Nikon\Ti2-SDK\bin` to the Micro-manager install directory, as described [here](https://micro-manager.org/NikonTi2).

* Install another copy of the recommended [Micro-Manager nightly build](https://download.micro-manager.org/nightly/2.0/Windows/) in the `C:\Program Files\Micro-Manager-2.0_YYYY_MM_DD_2` directory. This copy of Micro-manager will be used by the headless light-sheet acquisition engine.

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

* TODO: Write mantis test, some of which will test the microscope hardware

* TODO: Compose an acquisition that can test the microscope hardware without requiring a sample

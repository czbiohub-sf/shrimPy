## Overview - unofficial
shrimpy microscope is a fast multimodal microscope capable of acquiring label-free and fluorescence measurments simultaneously via two remote-refocus based arms that quickly acquire volumetric information. 

---
## Table of contents
- [Overview](#overview)
- [Table of contents](#table-of-contents)
- [Setup incubation chamber](#setup-incubation-chamber)
    - [OkoLab CO2, humidifier, and temperature control](#okolab-co2-humidifier-and-temperature-control)
        - [CO2](#co2)
        - [Humidifier](#humidifier)
        - [Temperature probe](#temperature-probe)
- [Sample Preparation](#sample-preparation)
  - [Apply silicone oil to the plate/slide](#apply-silicone-oil-to-the-plateslide)
- [Setting up the acquisition using `shrimpy Acquisition Engine`](#setting-up-the-acquisition-using-shrimpy-acquisition-engine)
      - [Micromanager with Label-free configuration](#micromanager-with-label-free-configuration)
      - [Micromanager with Light-sheet configuration](#micromanager-with-light-sheet-configuration)
  - [Setup autofocus](#setup-autofocus)
  - [YAML file:](#yaml-file)
  - [Defining XYZ positions:](#defining-xyz-positions)
    - [For HCS:](#for-hcs)
    - [For 8-well Ibidi Plate or other custom slides:](#for-8-well-ibidi-plate-or-other-custom-slides)
- [Run the acquisition](#run-the-acquisition)
- [FAQ](#faq)

---

## Setup incubation chamber
#### OkoLab CO2, humidifier, and temperature control
1. Press and hold the okolab monitor button to switch on/off
2. Check that the proper temperature, gas flow, and humidity parameters are set for the expereiment. The `search` icon has better view of all the connected components to the controller (i.e incubator, objective heater, humidity, airflow).
###### CO2
3. Open the CO2 valve located on the `hummingbird` microscope frame.
4. Check the CO2 levels and tune the gas flow nobs on the gas machine.
###### Humidifier
5. Check the water level on the humidifier and place it on the optical table.
6. Hook up the humidifier tube to the left side golden port of the chamber.

###### Temperature probe
7. Pass the thermometer probe through the side hole of the chamber. 
8. Place the probe in one of the wells. Make sure it is dipped inside.
9. Use kapton tape to fix the probe to the side of the chamber. Make sure the probe does not obstruct the FOV for imaging other wells.
---
## Sample Preparation
### Apply silicone oil to the plate/slide
1. Using the `1000cst` (most viscous solution) and the makeup palette, apply a layer of oil to the bottom of the plate/slide. 

**Note: This layer should have minimal number of bubbles and enough oil for autofocus to engage. Make sure to wear gloves and discard them before touching any other microscope or computer part. Handling the oil can be messy and we want to keep things clean. 

---
## Setting up the acquisition using `shrimpy Acquisition Engine`
1. Initialize two instances of micromanager to prepare both arms for acquisition. 
   1. Micromanager *230312* version with  `shrimpy-LF.cfg` configuration.
   2. Micromanager *230312 Nightly Build* version with `shrimpy-LS.cfg` configuration.
##### Micromanager with Label-free configuration
2. Using the micromanager with the `label-free` configuration, set `Channel LS` property group to `External Control`. This will reset the microscope to the default configurations including the filter cube to position 6. Then, set the `Imaging Path` propety group to `Epi`. Set the `Channel - LF` property group to `BF`. 
3. Focus the sample and check the focus is ~4700um.
4. Engage autofocus. [For more details](#setup-autofocus)
5. Switch to the `Imaging Path` to the `label-free` path and use the `KDC101` to adjust the position of `O2` and match the focal planes.
6. Inspect the fluorescence channels by switching the `Channel LS` property group to the respective fluorophore name or laser line (i.e GFP, TXR, Cy5, mCherry, etc)
##### Micromanager with Light-sheet configuration
1. Using the micromanager with the `light-sheet` configuration use the `KDC101` to adjust the position of `O3` to match the focal planes. 
2. Set-up the `Prime BSI` settings to:
   1. Readout -> 200MHz 
   2. Gain -> Sensitivity
3. Check the ROI to be captured and offset by changing the Y-coordinate. Click `ImageJ->Edit->Selection->Specify` and set the dimensions:
   1. Height: 2048
   2. Width: 256-300
   3. X-coordinate: 0 
   4. Y-coordinate: 896

** Note: this view is 3D with the coverslip in the x-orientation. 

### Setup autofocus
Check the LED on the microscope body and find the PFS button on the microscope focus knob.
1. One click == Turn on autofocus
   1. Autofocus **LED ON** + **one beep** = autofocus engaged 
   2. After the **beep**, turning the focus know changes the autofocus offset
2. Toggle autofocus state by clicking PFS button.

** Note: when it autofocus engages, the beep can be easily missed. 

### YAML file:
1. Copy the default [yaml file](../shrimpy/acquisition/settings/example_acquisition_settings.yaml) or one from a previous acquisition similar to the planned acquisition into the acquisition folder
2. Go over the YAML file and change the appropriate parameters.
3. Most likely one will change the parameters for:
   - ROI
   - timepoints and frequency
   - Channel names and exposures
### Defining XYZ positions:
1. Open the position manager and check the axis to store
   1. Z-drive (objective stage) and XY Stage position
2. Create a custom grid with position list manager.
3. Manually navigate to the center of the well of interest.
4. Generate the grid (i.e 3x3)
   1. Use that position as the center.
   2. Select the grid size with `+/-` sign
   3. Add the desired offset. Negative offsets == non-overlapping

#### For HCS:
4. Use the HCS generator under `Plugins -> Acquisition Tools -> HCS Site Generator` to image all the wells. 
5. Use the 3 point interpolation to adjust objective z-position across the plate.
 ---

#### For 8-well Ibidi Plate or other custom slides:
6. Use the HCS generator as above to `Create Custom` plate format
7. The 8-well Ibidi Plate can be found in the `Documents` folder. 

## Run the acquisition
1. Open powershell
2. Activate enviroment
    -`conda activate shrimpy`
3. Navigate to the acquisition folder:
    -`cd /path/to/acquisition` 
4. Run the acquisition engine
    -`shrimpy run-acquisition --help` for instructions.

---
## FAQ
1. _ The objective won't move up with the focus knobs_
   - Most likely the objective is on `escape` move and the `escape` button on the microscope body is on (green). Press and hold to disable. This ensures we can manually raise/lower the objective and avoid creating bubbles with the focus knob.

  

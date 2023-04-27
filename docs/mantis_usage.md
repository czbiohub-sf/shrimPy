## Table of contents
- [Table of contents](#table-of-contents)
- [Overview](#overview)
- [Setup incubation chamber](#setup-incubation-chamber)
    - [OkoLab temperature and humidifier](#okolab-temperature-and-humidifier)
      - [CO2 Supply](#co2-supply)
      - [Temperature probe](#temperature-probe)
- [Acquisition setup with Mantis Acquisition Engine](#acquisition-setup-with-mantis-acquisition-engine)
  - [Sample focus evaluation using the Epi-port](#sample-focus-evaluation-using-the-epi-port)
  - [Setup autofocus](#setup-autofocus)
  - [Setup Label-free arm](#setup-label-free-arm)
  - [Setup Light-sheet arm](#setup-light-sheet-arm)
    - [Centering and cropping the ROI](#centering-and-cropping-the-roi)
    - [Prime BSI Camera](#prime-bsi-camera)
  - [YAML file](#yaml-file)
  - [Defining xy positions](#defining-xy-positions)

## Overview

## Setup incubation chamber
#### OkoLab temperature and humidifier
Press and hold the okolab monitor button to switch on/off
The `settings` window has better view of the individual connected components (i.e incubator, objective heater, humidity, etc)
Hook up the humidifier tube to the left side golden port of the chamber.

##### CO2 Supply
Open CO2 valve.
Check CO2 levels 

##### Temperature probe
Pass the thermometer probe through the side hole of the chamber. 
Place the probe in one of the wells. 
Use the kapton tape to fix the probe.
## Sample Preparation
### Oil
1000cst
nice layer of oil for the autofocus

## Acquisition setup with Mantis Acquisition Engine
For setup open two micromanager instances:
one is the latest nightly build and the other is 09.20 version
### Sample focus evaluation using the Epi-port
Look at the sample in the epi port and focus the label free
Check the sample is at ~4700um. We can offset with the piezo. 
Filter cube in position number 6

### Setup autofocus
One click to turn it on.
Look at the microscope body LED.
LED no blinking == Autofocus is on
Blinking LED  == Offeset?

Note: holding the `escape` button in the microscope allows one to disable this feature. This ensure we can manually raise/lower the objective and avoid creating bubbles

### Setup Label-free arm
Check Kohler illumination
### Setup Light-sheet arm
Focus control 
Voltages for the Galvos for positioning the sample

#### Centering and cropping the ROI
2048 x 256 
#### Prime BSI Camera
Looking at the 
Settings -200MHZ
Gain "Sensitivity

### YAML file
Go over the YAML file
Change the parameters for:
timepoints, volume , ROI,
### Defining xy positions
We can create a custom grid with position list manager
Use the 3 point interpolation to generate a plate z-objective map. 






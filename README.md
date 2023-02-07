# mantis_automation

## Installation

mantis depends on following packages:

* Acquisition: pycro-manager (to control micro-manager), recOrder-napari, nidaqmx
* Analysis: dexp, recOrder-napari
* io and visualization: iohub, napari

### Apple Silicon (M1)

```shell
conda create -n mantis python=3.9
conda activate mantis  
pip install nidaqmx napari pyqt6 pycromanager dexp
```

Now clone `recOrder` in a separate folder and install it:

```shell
git clone https://github.com/mehta-lab/recOrder
pip install -e <path to recOrder>
```

Similarly clone `iohub` and install:

```shell
git clone https://github.com/czbiohub/iohub
pip install -e <path to iohub>
```

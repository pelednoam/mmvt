<img src=https://user-images.githubusercontent.com/35853195/42889397-52f9c75e-8a78-11e8-9da8-86ccc3a30a80.png align="right" hight=120 width=120/>


## Multi-Modality Visualization Tool

The visualization and exploration of neuroimaging data are important for the analysis of anatomical and functional images and statistical parametric maps. While two-dimensional orthogonal views of neuroimaging data are used to display activity and statistical analysis, real three-dimensional (3D) depictions are helpful for showing the spatial distribution of a functional network, as well as its temporal evolution. For our best knowledge, currently, there is no neuroimaging 3D tool which can visualize both MEG, fMRI and invasive electrodes (ECOG, depth electrodes, DBS, etc.). Here we present the Multi-Modality Visualization Tool (MMVT). The tool was built for researchers who wish to have a better understanding of their neuroimaging anatomical and functional data. The true power of the tool is by visualizing and analyzing data from multi-modalities. MMVT is built as two separated modules: The first is implemented as an add-on in 'Blender”, an open-source 3D visualization software. The add-on is an interactive graphical interface which enables to visualize functional and statistical data (MEG and/or fMRI) on the cortex and subcortical surfaces, invasive electrodes activity and etc. The tool can also be used for a better 3D visualization of the anatomical data and the invasive electrodes locations. The other module is a standalone software, for importing and preprocessing. The users can select the data they want to import to Blender and how they want to process it.

The module supports many types of analyzed data:
* FsFast (FreeSurfer Functional Analysis Stream)
* SPM (Statistical Parametric Mapping)
* MNE (a software package for processing MEG and EEG)
* MEG raw data (fif files)
* FieldTrip (MATLAB software toolbox for neuroimaging analysis)


The users can also reprocess raw data using wrappers for FaFast and MNE-python (a python package for sensor and source-space analysis of MEG and EEG data).

<a href="https://doi.org/10.5281/zenodo.438343"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.438343.svg" alt="DOI"></a>

## New Website
Please visit our new website: [mmvt.org](http://mmvt.org)

## Videos & Figures
* The [aparc.DKTatlas](https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation) FreeSurfer atlas:
![Example](https://user-images.githubusercontent.com/1643819/39079174-8b61dc1a-44e3-11e8-8ce6-1c783596d1ae.png)

<!--- * Spatial and temporal ttest result of MEG activation
 ![Example](https://cloud.githubusercontent.com/assets/1643819/17341466/c1ac0548-58c2-11e6-9736-a85163f80521.gif "spatial and temporal ttest result of MEG activation") -->
 * Short demo how to use the main features (click on the image):
 
 <a href="https://www.youtube.com/watch?v=vPD4DorhMgA&t=9s" target="_blank"> <img src="https://img.youtube.com/vi/yBba7f12GmQ/0.jpg" alt= "mmvt demo"> </a>
* Resting State fMRI & Cortical labels correlation:
![Example](https://cloud.githubusercontent.com/assets/1643819/24374566/5ce2dce4-1303-11e7-9b3a-c23448e5114e.gif)
* Spatial and temporal ttest result of MEG activation:
![Example](https://cloud.githubusercontent.com/assets/1643819/17341466/c1ac0548-58c2-11e6-9736-a85163f80521.gif)
<!-- * MEG & Electrodes & Coherence
![Example](https://cloud.githubusercontent.com/assets/1643819/17341742/03e0af80-58c4-11e6-8587-125cde58e6b8.gif "MEG & Electrodes & Coherence") -->
<!--* Inflating Brain
![inflating_meg](https://user-images.githubusercontent.com/1643819/32626655-f58758be-c55d-11e7-94c6-de246c291905.gif) -->

<!--## Tutorials
![Example](https://cloud.githubusercontent.com/assets/1643819/17341371/4d3505de-58c2-11e6-8bae-91165c573a07.gif "MEG-fMRI-electrodes example") -->

## Features

A list of features can be found [here](https://github.com/pelednoam/mmvt/wiki/MMVT-features)

The features can be seen [here](https://www.youtube.com/watch?v=vPD4DorhMgA&t=9s)

## Installation
**Windows** [full guide](https://github.com/pelednoam/mmvt/wiki/Windows-installation-full-guide)

**Linux & Mac OS** [full guide](https://github.com/pelednoam/mmvt/wiki/Linux-and-OSX-Installation)

The tool itself can run on windows, mac and linux.
In the preprocessing pipeline there are several calls to [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (runs only on linux/mac). Beside of that, you can use the tool also on windows.

**Download the template brain Colin27 to learn more about the tool**

## Template brain and data
We've imported [colin27](http://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27Highres) into MMVT, and included data we morphed from a patient. The data includes recording from EEG, MEG, fMRI, and sEEG for the [MSIT](https://www.nature.com/articles/nprot.2006.48) task. The data can be downloaded from [here](https://www.dropbox.com/s/hpt5t9gt8migna8/colin27.zip?dl=0) (1GB). Extract the zip file in the mmvt_blend folder which was created in your mmvt_root folder. Then, open Blender, close the splash screen and open (File->open) colin's blend file (colin27_laus125.blend).

* To learn more about the tool you can find several self-explanatory tasks for Colin27 data [here](https://github.com/pelednoam/mmvt/wiki/Colin27-exercise).
   - Step by step answers for the tasks can be found [here](https://docs.google.com/document/d/1FD2vA_eSbGMsZIxZs_8wAstyNN5QMYIxkV35jaj2qyM/edit?usp=sharing)


## Blender shortcuts
* Rotate the brain using the middle mouse button.
* Select objects (electrodes / sensors / cortical lables) using the right mouse button. To select cortical labels you need to change first the view to "ROIs" in the Appearence panel.
* Zoom in/out using the mouse scrolling.

## Preprocessing
The preprocessing tutorial can be found in the [wiki sidebar](https://github.com/pelednoam/mmvt/wiki).

## Update the code
**Linux**: The program can be updated without using the "git pull" function.
* Lunch MMVT.
* Open the "Import objects and data" panel.
* Press the "Update MMVT" button at the top of the panel.

**Windows**:
* In the Git CMD terminal chage the directory to the mmvt_code folder (example: cd c:\Users\Jhon\mmvt_root\mmvt_code)
* Type: git pull


## Contributors
- Noam Peled (npeled@mgh.harvard.edu)
- Ohad Felsenstein (ohad.felsenstein@biu.ac.il)

## Acknowledgments
This research was partially funded by the Defense Advanced Research
Projects Agency (DARPA) under Cooperative Agreement Number
W911NF-14-2-0045 issued by ARO contracting office in support of [DARPA's
SUBNETS Program](https://transformdbs.partners.org/). The views, opinions, and/or findings expressed are
those of the author and should not be interpreted as representing the
official views or policies of the Department of Defense or the U.S.
Government.  This research was also funded by the NCRR (S10RR014978) and NIH (S10RR031599,R01-NS069696, 5R01-NS060918, U01MH093765).

## Suggested Citation
N. Peled and O.Felsenstein et al. (2017). MMVT - Multi-Modality Visualization Tool. GitHub Repository. https://github.com/pelednoam/mmvt DOI:10.5281/zenodo.438343

## Licensing

MMVT is [**GNU GPL licenced (v3.0)**](https://github.com/pelednoam/mmvt/blob/master/LICENSE)

 

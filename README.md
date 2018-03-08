<img src=https://user-images.githubusercontent.com/1643819/36446044-0ba7e0b4-164e-11e8-8ff8-f2963bcf78be.png align="right"/>
<br><br>

## Multi-Modality Visualization Tool

The visualization and exploration of neuroimaging data are important for the analysis of anatomical and functional images and statistical parametric maps. While two-dimensional orthogonal views of neuroimaging data are used to display activity and statistical analysis, real three-dimensional (3d) depictions are helpful for showing the spatial distribution of a functional network, as well as its temporal evolution. For our best knowledge, currently, there is no neuroimaging 3d tool which can visualize both MEG, fMRI and invasive electrodes (ECOG, depth electrodes, DBS, etc.). Here we present the multi-modality visualization tool (MMVT). The tool was built for researchers who wish to have a better understanding of their neuroimaging anatomical and functional data. The true power of the tool is by visualizing and analyzing data from multi-modalities. MMVT is built as two separated modules: The first is implemented as an add-on in 'Blender”, an open-source 3d visualization software. The add-on is an interactive graphics interface which enables to visualize functional and statistical data (MEG and/or fMRI) on the cortex and subcortical surfaces, invasive electrodes activity and so on. The tool can also be used for a better 3d visualization of the anatomical data and the invasive electrodes locations. The other module is a standalone software, for importing and preprocessing. The users can select the data they want to import to Blender and how they want to process it. The module support many types of analyzed data, like FsFast (FreeSurfer Functional Analysis Stream) and SPM (Statistical Parametric Mapping) for fMRI, MNE (a software package for processing MEG and EEG) raw data for MEG and FieldTrip (MATLAB software toolbox for neuroimaging analysis) data structures for the invasive electrodes. The users can also reprocess raw data using wrappers for FaFast and mne-python (a python package for sensor and source-space analysis of MEG and EEG data).

<a href="https://doi.org/10.5281/zenodo.438343"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.438343.svg" alt="DOI"></a>

## Videos

<!--- * Spatial and temporal ttest result of MEG activation
 ![Example](https://cloud.githubusercontent.com/assets/1643819/17341466/c1ac0548-58c2-11e6-9736-a85163f80521.gif "spatial and temporal ttest result of MEG activation") -->
* Resting State fMRI & Cortical labels correlation
![Example](https://cloud.githubusercontent.com/assets/1643819/24374566/5ce2dce4-1303-11e7-9b3a-c23448e5114e.gif)
* Spatial and temporal ttest result of MEG activation
![Example](https://cloud.githubusercontent.com/assets/1643819/17341466/c1ac0548-58c2-11e6-9736-a85163f80521.gif)
<!-- * MEG & Electrodes & Coherence
![Example](https://cloud.githubusercontent.com/assets/1643819/17341742/03e0af80-58c4-11e6-8587-125cde58e6b8.gif "MEG & Electrodes & Coherence") -->
<!--* Inflating Brain
![inflating_meg](https://user-images.githubusercontent.com/1643819/32626655-f58758be-c55d-11e7-94c6-de246c291905.gif) -->

<!--## Tutorials
![Example](https://cloud.githubusercontent.com/assets/1643819/17341371/4d3505de-58c2-11e6-8bae-91165c573a07.gif "MEG-fMRI-electrodes example") -->


## Installation
Please go through the installation in this wiki [page](https://github.com/pelednoam/mmvt/wiki/Installation).

The tool itself can run on windows, mac and linux.
In the preprocessing pipeline there are several calls to [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (runs only on linux/mac). Beside of that, you can use the tool also on windows.

## Template brain and data
We've imported [colin27](http://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27Highres) into MMVT, and included data we morphed from a patient. The data includes recording from EEG, MEG, fMRI, and sEEG for the [MSIT](https://www.nature.com/articles/nprot.2006.48) task. The data can be downloaded from [here](https://tinyurl.com/yab2vfzx) (1GB). Extract the zip file in the mmvt_blend folder which was created in your mmvt_root folder. Then, open Blender, close the splash screen and open (File->open) colin's blend file (colin27_laus125.blend). After that, you'll need to press space anywhere on the screen, write "run mmvt" and press enter. We've also written several self-explanatory tasks for this data, that can be found [here](https://tinyurl.com/yd2axzsg) (feel free the add comments).
Few Blender shortcuts you should know:
* Rotate the brain using the middle mouse button.
* Select objects (electrodes / sensors / cortical lables) using the right mouse button. To select cortical labels you need to change first the view to "ROIs" in the Appearence panel.
* Zoom in/out using the mouse scrolling.

## Preprocessing
The preprocessing tutorial can be found in the [wiki](https://github.com/pelednoam/mmvt/wiki/Preprocessing).

## Contributors
- Ohad Felsenstein (ohad.felsenstein@biu.ac.il)
- Noam Peled (npeled@mgh.harvard.edu)

## Acknowledgments
This tool is being developed as part of the [Transform DBS project](https://transformdbs.partners.org/).
It's sponsored by:
- The U.S. Army Research Office and Defense Advanced Research Projects Agency under Cooperative Agreement Number W911NF-14-2-0045. 
- The NCRR (S10RR014978) and NIH (S10RR031599,R01-NS069696, 5R01-NS060918, U01MH093765)

## Suggested Citation
O.Felsenstein and N. Peled et al. (2017). MMVT - Multi-Modality Visualization Tool. GitHub Repository. https://github.com/pelednoam/mmvt DOI:10.5281/zenodo.438343

## Licensing

MMVT is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2016, authors of MMVT
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of MMVT authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**

# Corrosion Assessment and 3D Visualization of Hydraulic Steel Structures

This repository provides selected research materials accompanying the manuscript **“A closed-loop edge-server collaborative intelligence framework for corrosion assessment and 3D visualization of hydraulic steel structures.”**

The release is intended to help readers examine key implementation details and access a subset of the corrosion data. It does not contain the complete dataset or the full edge–server system.

## Contents

### 1. Corrosion segmentation model

- Source code for the **dual-branch spatial–frequency module** [**dsfm.py**].
- The YAML configuration file for the **YOLO26-CorroSeg backbone** [**YOLO26-CorroSeg.yaml**].

Together, these files document the released model component and its backbone configuration.

### 2. Corrosion dataset subset

A subset of images of corroded hydraulic steel structures and their corresponding annotations is provided for inspection and testing. The complete project dataset is not included because of confidentiality restrictions.

- **Dataset location:** `[Google Cloud Drive (314MB)]`  ：[https://drive.google.com/file/d/1xpMcc4sXuCSKoAqmjJ7mM08H3afWkK8F/view?usp=drive_link]

- **Number of released images:** `[35]`  
- **Annotation format:** `[YOLO-Seg]`  
- **Class definitions:** `[corrosion]`

### 3. Windows 3D Gaussian Splatting materials

The repository includes materials for configuring the 3D Gaussian Splatting environment with Anaconda on Windows and the corresponding video-based training script.

- **A compressed archive of the Anaconda environment used to run 3DGS on Windows:** `[Google Cloud Drive (2.95GB)]`  ：https://drive.google.com/file/d/1cGCyGK59yy9q0mbRakBBri8x9kvhdJO9/view?usp=sharing

- **Training script:** `Once the 3D Gaussian Splatting (3DGS) environment has been configured, simply place the '**train_video.py**' script into the root directory of the gaussian-splatting project[https://github.com/graphdeco-inria/gaussian-splatting], and 3D reconstruction from a video can be performed in a single step.

Before running the script, update its input and output paths to match your local directory structure. See the files in this section for the required environment configuration and script parameters.

## Availability and scope

This repository contains selected materials from the study. Other project data and components of the full system are not publicly released because of research project confidentiality requirements. Accordingly, the repository should not be interpreted as a complete reproduction package for every experiment reported in the manuscript.

## Citation

If you use these materials, please cite the associated manuscript:

**“A closed-loop edge-server collaborative intelligence framework for corrosion assessment and 3D visualization of hydraulic steel structures.”**  
`[ AUTHORS, JOURNAL, YEAR, AND DOI WHEN AVAILABLE]`

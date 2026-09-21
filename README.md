# 3DBodyCircumferenceEvaluator

Proof of Concept (PoC) to validate the application of technologies for a physical evaluator responsible for the calculation of corporal circumferences through deep 3D cloud points data.

## Overview

The human body is an amazing machine, capable of overcoming limits through intense and dedicated effort. However, to optimize performance and prevent setbacks, precise, objective monitoring is essential. Fatigue, injury and inefficient training routines are often the result of failing to accurately track physical changes. 3D Body Circunference Evaluator addresses this critical need by providing a innovative method for reliable, high-precising body metrics.

First Prototype (Measures in Chest, waist and hip):
<img src="/img/first_measuring_prototype.png" alt="first_prototype" width="800"/>

Second Prototype (More measures added):
<img src="/img/second_measuring_prototype.png" alt="first_prototype" width="800"/>

## Game Setup 𝄂𝄂—𝄂𝄂

These are the used equipments to the execution of the experiment:

Orbbec Astra 2 RGB-D Sensor
<img src="/img/orbbec_astra2.jpeg" alt="Orbbec Astra 2" width="400"/>

BKL Turntable
<img src="/img/bkl_turntable.jpeg" alt="Rotating Turntable" width="400"/>

Sensor and turntable connected
<img src="/img/sensor_and_turntable.jpeg" alt="Rotating Turntable" width="400"/>

## Highlights

Main menu
<img src="/img/main_screen.png" alt="Main Screen" width="800"/>

User window display with threshold range and restriction zones
<img src="/img/screenshot.png" alt="Main Screen" width="800"/>

Point cloud after RGB and D images generation and ICP alignment
<img src="/img/generated_point_cloud.png" alt="Main Screen" width="800"/>

Reconstructed triangle mesh ready for landmarks detection and posterior body measurement in high detail
<img src="/img/reconstructed_mesh.png" alt="Main Screen" width="800"/>

First version of the measurement with 3 landmarks: Chest, waist and hip
<img src="/img/measurement_first_version.png" alt="Main Screen" width="800"/>


## Marathon Route 👣🏃
- [x] RGB-D camera choosing (Firstly I decided to choose the Intel Realsense D435i first)
- [X] Intel Realsense D435i setup and test
- [X] Familiarization with RGB-D (Intel Realsense D435i) camera: Generate .ply file and understand of point cloud basic properties
- [X] Attempt to generate a 3D reconstruction from a small physical scenario (bedroom) with this camera (Through the https://www.open3d.org/docs/release/tutorial/reconstruction_system/capture_your_own_dataset.html toolkit)
- [X] Receipt of the Orbbec Astra 2
- [X] Orbbec Astra 2 setup and test
- [X] Generation of .ply files with Orbbec Astra 2 through the application OrbbecViewer to familiarize with the available filters, modes, options...
- [X] Structure studying and execution of the script from the Orbbec API to generate .ply directly with Orbbec Astra 2
- [X] Modification in the Orbbec Astra 2 point cloud generation and user window display scripts to deal with filters: Hole filling, threshold to determine desired regions 
- [X] Manual testing of various rotation and translation values for generation of geometric transformation to 3D reconstruction using turntable (At the beginning, the point cloud alignment was made using transformation methods very before the application of point clouds alignment algorithms) 
- [ ] Lightning profiles to correct glare and shadow regions to after avoid color variations during ICP iterations 
- [X] Threading to execute several processes in real time: camera display, turntable, RGB-D sensor registration...
- [X] Turntable control through serial messaging: Stop, rotate, speed control, constant state checking during threading process 
- [X] Counter implementation to regressive countdown performing until the scanning start
- [X] Generation of RGB and thresholded depth aligned frames with each person rotation:
- [X] Mask Operations:
    - [X] Background removal and let only the person that is in front of the sensor through neural network
    - [X] Corners smoothing
    - [X] Removal of invalid pixels
- [X] Depth Image Operations:
    - [X] Masking of invalid pixels
    - [X] Discontinuities detection
    - [X] Bilateral filtering to reduce "depth bleeding" pixels
    - [X] Flying pixels removal
    - [X] Erosion to remove remaining "depth bleeding" pixels
- [X] Point clouds generation through RGB+D images after image processing
- [X] Application of cropping to remove undesired sub point clouds in each registered point cloud set
- [X] Application of ICP (Iterative Closest Point) algorithm to align point clouds through their geometric correspondences
- [X] Triangle mesh generation using the whole reconstructed point cloud using Poisson method
- [ ] Mesh rendering for high resolution/realism
- [X] Body landmarks detection in triangle mesh
    - [X] Detection of chest, waist and hip
    - [X] Detection of 37 landmarks across torso, arms and legs
- [X] Measures extraction from each detected landmark grouping
- [ ] Release of version v0.1.0
... 

TBD the rest and more details


## Before enter in the game... 👟

Be sure that you have this footage:
* Python 3.12 or higher
* Windows 10 or higher or Linux
* Pip or Conda package installer
* Intel RealSense SDK
* ...

### Defining the field game through pip virtual environment (venv)

1. Create the environment by running:
```
python3 -m venv .3Dbce_env
```
2. Activate it:
```
source .3Dbce_env/bin/activate
```
3. After, install all the requirements that will be used:
```
pip install -r requirements.txt
```
4. Verify that the new environment was installed correctly:
```
pip list
```

### Defining the field game through conda virtual environment

1. Create the environment from the `3Dbce_env.yml` file:
```
conda env create -f 3Dbce_env.yml
```

2. Activate it:
```
conda activate 3Dbce_env
```

3. Verify that the new environment was installed correctly:
```
conda env list
```
or by `pip list`

### Installing directly the requirements.txt file (Global without any isolation from your work environment): 
`pip install -r requirements.txt`
Obs: It's possible but not recommended due the fact that packages can conflict with each other case different versions from the same package be installed.

<!-- ## Generation of the Game Statistics (Dataset)

It was set a Orbbec Astra 2 sensor for the acquisition of RGB-D images from the person motion, the person is positioned on a rotating base in clockwise sense of rotation for the acquisition of the 3D cloud point, this data will be used after to calculate the metrics. -->

<img src="/img/filmed_person.png" alt="filmed_person" width="800"/>

# DSA-Thread-Sampling

* Thread Sampling is the script that can generate images with annotation masks and Thread sampling
* Data is in the same format as the JIGSAWS-Cogito-Annotations repo

# File Structure
Tasks can be Needle_Passing, Knot_Tying, Suturing
Each task has its own data inputs and output folder, under which they are ordered by the usual ```<Task>_<Subject>_<Trial>```
* ```<task>```
    * annotations_pre
    * images_pre
    * kinematics
    * motion_primitives_combined
    * motion_primitives_combined
    * output
        * ```<Task>_<Subject>_<Trial>```
            * frame_0001.png
    * output_ext
    * transcriptions

# Scripts

## Thread_Sampling.py

* JSONInterface: Helps to extract polygons, keypoints, and polylines from Annotaiton JSON files
* Iterator: loops through all images and draws segmentation annotaiton

## Thread_Sampling.py

* JSONInterface: Helps to extract polygons, keypoints, and polylines from Annotaiton JSON files
* Iterator: loops through all images and draws segmentation annotaiton

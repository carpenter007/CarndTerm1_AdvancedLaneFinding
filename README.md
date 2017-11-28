## Advanced Lane Finding

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.
The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.
The images in `test_images` are for testing the pipeline on single frames.
Examples of the output from each stage of your pipeline in the folder is stored in `ouput_images`
The 'writeup.md' gives a detailled description of how and why each step is processed.
The video called `project_video.mp4` is the video on which my pipeline works well.
The `challenge_video.mp4` video is an extra (and optional) challenge to test the pipeline under somewhat trickier conditions.
The `harder_challenge.mp4` video is another optional challenge and is brutal!


This project is based on Udacitys Self-driving car Nanodegree program term 1 project: https://github.com/udacity/CarND-Advanced-Lane-Lines
Thanks to Sebastian Thrun, David Siller, Rajesh (Mentor), Paul (Reviewer) for the create lectures, projects, support and discussions.

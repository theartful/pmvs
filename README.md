# Patch-based Multi-view Stereo
This repo contains our attempt at implementing the paper "Accurate, Dense,
and Robust Multi-View Stereopsis" by Yasutaka Furukawa and Jean Ponce.
This algorithm is highly parallelizable, which makes golang a suitable
language to implement it on. My main focus of this project is to learn go,
and concurrent programming.

## Algorithm Overview
The algorithm consists of three phases: initial matching, expansion,
and filtering.

### Initial Matching
In this phase a sparse set of patches is generated. This is done by first
detecting corner and blob features in each image. Two detectors are used:
Difference of Gaussians, and Harris. Then each feature in each image is
matched with other candidates that lie near the epipolar line corresponding
to the feature. Then we triangulate to find the 3D point associated with
the pair, and use these points as initial values for the centers of patches
to be created. An optimization routine is then run on these patches to
maximize photometric consistency.

## Current State
Feature detection and initial matching are implemented. This leaves the
expansion and filtering steps. 

## TODO
- [ ] Detect better features
- [ ] Make the code run concurrently
- [ ] Implement the expansion step
- [ ] Implement the filtering step

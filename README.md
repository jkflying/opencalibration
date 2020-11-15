![CI Status](https://github.com/jkflying/opencalibration/workflows/C/C++%20CI/badge.svg)

# opencalibration
An extremely fast camera intrinsic and extrinsic calibration library designed to scale.

Based on the concept of staged pipelines, where chunks can work in parallel, keeping all resources (IO, FPU, POPCNT, memory bandwidth) busy through the entire processing, while still keeping reproducible outputs across any machine run with the same parameters.

Still in early stages, check back later unless you're just looking for ideas.

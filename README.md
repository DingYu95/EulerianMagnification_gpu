# Eulerian motion magnification

This is a quick implantation of [paper](http://people.csail.mit.edu/mrub/papers/vidmag.pdf), using image pyramids to amplify motion or color variations in videos more discernible. And their later work, using steerable pyramids gives the coolest result I've ever seen for computer vision before the rise of deep learning. (It still is. :) And I'm working on a real-time implementation of that as well.

Pyramids used in this paper can be readily built from function available in OpenCV for both CPU and GPU version, which gives a good opportunity to try gpu-accelerated code. Please check [notes](notes.md) for details.

## Code structure:
As described in paper, there are mainly two part in the magnification process. Spatial decomposition, and temporal filtering.

In spatial decomposition, pyramids (Gaussian/Laplacian) are built from incoming images. Then, use temporal filters like IIR, or chosen ideal frequency manually.

## TODOs:
- [ ] Make code objected-orientated, which is more convenient to do pre-allocation. And build a pipeline of pyramids instances with its own member functions.
- [ ] Add threads for different stage in pipeline.
- [ ] With threads enabled, process different channels in multiple threads, which might give huge performance enhancement.
- [ ] Add more temporal filters, like butterworth and wavelet, as mentioned in paper.
- [ ] Better docs for functions.

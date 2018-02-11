# CUDA-Connected-Component-Labelling

Connected-component labeling (alternatively connected-component
analysis, blob extraction, region labeling, blob discovery, or region
extraction) is an algorithmic application of graph theory, where subsets of
connected components are uniquely labeled based on a given heuristic.

Connected-component labeling is used in computer vision to detect
connected regions in binary digital images, although color images and data
with higher dimensionality can also be processed.

For example, in problems where we have images like these -
![Alt text](http://members.cbio.mines-paristech.fr/~nvaroquaux/formations/scipy-lecture-notes/_images/plot_synthetic_data_1.png "Images with connected regions")

When integrated into an connected component labeling can operate on a variety of information. B
image recognition system or human-computer interaction interface,lob
extraction is generally performed on the resulting binary image from a
thresholding step.

Blobs may be counted, filtered, and tracked.
Considering the prominence of Connected Component Labelling and its
applications, an approach towards speeding up its execution has been made
by a parallel GPU based implementation.

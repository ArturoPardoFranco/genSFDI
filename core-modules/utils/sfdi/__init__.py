'''
Organization of the SFDI processing module
Arturo Pardo
Grupo Ing. Fotónica, 2018

LITERATURE ---- ---- ---- ----

We will base most of the SFDI module out of the following articles:
(1) Cuccia, Tromberg, et al., 2009
(2) Cuccia, Tromberg, et al., 2005

ORGANIZATION ---- ---- ---- ----

Code:
( ) - To be implemented
(W) - Work in Progress (WIP)
(!) - Some problems found
(X) - Done

The following modules need implementation:

[X] A module that, by means of the diffusion approximation, estimates the MTF of
    any specific material. This is required originally for MTF calibration using
    Spectralon (or any other reference material, for that matter).
    [diffusionMTF.py]

[W] A module that, given a MTF reference (obtained by the previous section), gets
    the calibrated reflectance Rd(fx, lambda) for an MI image.
    [MTFCalibration.py]

[ ] A module that estimates absorption and reduced scattering (mu_a, mu_s') by means
    of the calibrated MTF (nonlinear least squares).
    [muCalculations.py]

[ ] A module that speeds up the previous module by using whatever is used by Cuccia
    in their 2009 article.
    [gridMuCalculations.py]


IMPLEMENTATION NOTES ---- ---- ---- ----

'''

from utils.sfdi.diffusionMTF import *

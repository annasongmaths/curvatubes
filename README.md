# Curvatubes - a curvature model for the generation of tubes

``Curvatubes`` is a Python code for generating tubular, membranous and porous 3D shape textures.\
The theoretical framework is described in the article (.bib citation below)
> [Generation of Tubular and Membranous Shape Textures with Curvature Functionals](http://10.28.35.11:8090/lab?token=24c64737bb0ba185416b19dd32fa10b4d4aa59fdbb6c4d0e)
> 
> Published 19 August 2021, Anna Song (Imperial College London and The Francis Crick Institute)

These shapes are modeled as optimizers of a curvature functional *F(S)* representing
a curvature-based geometric energy of surfaces. This functional generalizes the
classical Willmore and Helfrich energies, by allowing the principal curvatures
to play non-symmetric roles.

The original problem is approximated by a phase-field volumetric energy *Feps(u)*, which transposes the optimization on 2D surfaces *S* to 3D scalar fields *u*. This allows for an efficient and
flexible GPU algorithm, where topological changes encountered during the flow
are addressed seamlessly. The implementation benefits from the automatic differentiation
engine provided by PyTorch, combined to optimizers such as Adam and L-BFGS.

``Curvatubes`` leads to a wide continuum of shape textures, encompassing tubules and 
membranes of all sorts, such as porous anisotropic structures or highly branching networks. In particular, many more patterns than those presented in the publication can be generated.

## Updates

- 17/08/2022 : 
    - fixed some compatibility issues with the latest version of `torch.fft` in `pytorch` (beyond 1.8). The first released version of `curvatubes` was compatible with `pytorch` up to version 1.7. See [how the old and new versions behave](https://github.com/pytorch/pytorch/wiki/The-torch.fft-module-in-PyTorch-1.7).
    - fixed some compatibility issues with the latest version of `skimage.measure.marching_cubes`.
    - also fixed some minor things

## Acknowledgments

This work has been made possible thanks to the kind support of my supervisors,
[Anthea Monod](https://sites.google.com/view/antheamonod/home) (1) and [Dominique Bonnet](https://www.crick.ac.uk/research/find-a-researcher/dominique-bonnet) (2), as well as [Antoniana Batsivari](https://www.researchgate.net/profile/Antoniana-Batsivari) (2) who showed the images of bone marrow vessels which inspired this work.

(1) Department of Mathematics, Imperial College London \
(2) Haematopoietic Stem Cell Laboratory, The Francis Crick Institute

# Usage

This repository is organized as follows:

- `cvtub` : contains the curvatubes module
    - `curvdiags.py` : curvature diagram of a surface, i.e., histogram of its curvature statistics
    - `energy.py` : phase-field energy Feps
    - `filters.py` : GPU convolutional filters (Gaussian blur, differential operators: grad, Hess, div)
    - `generator.py` : leading function that generates shapes by optimizing Feps(u) on u with a H^{-1} flow
    - `utils.py` : useful auxiliary functinos
    - `__init__.py`
    
    
- `notebooks` : contains jupyter notebooks that demonstrate how to use curvatubes
    - `Curvatubes.ipynb` : usage of curvatubes, with numerical experiments from the article
    - `Willmore_flow.ipynb` : usage of a variant, for the L^2 flow in the special case of the Willmore energy
    
    
- `results` : will contain your future numerical outputs when you run the notebooks
    - `gallery` : example shapes related to those in the paper, with images and phase-fields u saved under .nii.gz format

The basic usage is to call ```from cvtub.generator import _generate_shape```, and wrap it\
into a more handy function as done in Curvatubes.ipynb. You are now ready to go!

**Each shape you create will be unique :)** \
If you obtain interesting shapes and want to show them to me, I will be delighted to know more about it.

# Academic use

If you use codes or ideas from this research paper, **please cite** it inside your work. Thank you !


```tex
@article{song_generation_2021,
  author = {Anna Song},
  title = {Generation of Tubular and Membranous Shape Textures with Curvature Functionals},
  journal = {Journal of Mathematical Imaging and Vision},
  year = {2021},
  month = aug,
  volume = {64},
  number = {1},
  pages = {17--40},
  doi = {10.1007/s10851-021-01049-9},
  url = {https://doi.org/10.1007/s10851-021-01049-9},
  publisher = {Springer Science and Business Media {LLC}}
}
```

# License

This code is licensed under the permissive [MIT License](https://en.wikipedia.org/wiki/MIT_License).

# Bugs and Feedback

If you have remarks or find a bug in the code, please contact me directly at a.song19@imperial.ac.uk.

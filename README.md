
# Canonical Holistic Morphometric Analysis (cHMA)

## Author:

#### Dr. Brian Anthony Keeling

## Installation:

Note: Please add the Github python script to your Python repository
first.

``` python
!pip install cHMA 
```

## Summary:

Canonical Holistic Morphometric Analysis (cHMA) is a novel methodology
to directly compare morphometric quantities across morphologically and
interspecifically variable bones such as trabecular bone. In other
words, this method allows for the direct analysis of entire structures
without the need for landmarks or corresponding shapes. The goal of cHMA
is to create a representative canonically averaged shape of all
individuals in a sample to serve as a reference to directly compare
morphometric quantities between variable bone shapes. This method is a
novel extension of Holistic Morphometric Analysis (HMA) which is
designed to qualitatively analyze morphometric quantities of whole bone
structures without the need of landmark-based approaches or subregions
of interest (Bachmann et al., 2022; Gross, Kivell, Skinner, Nguyen, &
Pahr, 2014; Steiner et al., 2021). While there have been various HMA
approaches (Bachmann et al., 2022), the general workflow has been to
standardize each trabecular image domain, create a volumetric mesh of
the trabecular bone, calculate the morphometric quantities from the bone
image, and interpolate these morphometric values on the generated mesh.
Morphometric values are sampled onto the mesh by creating the same 3D
background grid for each trabecular bone image in a sample. By using
overlapping sampling spheres, the morphometric values of the trabecular
bone are assigned to the 3D background grid and then interpolated onto
the overlapping mesh.

## Methodology:

![](/Figure1.jpg)

**Figure 1.** The canonical holistic morphometric analysis (cHMA)
process as described by Bachmann et al. (2022). Each of the four steps
highlighted in this process follow the procedures from top to bottom and
left to right with the next step starting at the top of the following
step.

The cHMA process extends the traditional HMA workflow by contributing a
canonical bone mesh which is registered to all trabecular images,
creating topological meshes with the same connectivity and vertices,
thus permitting the direct analysis of scalars by matching vertices
across individual meshes (Bachmann et al., 2022).

**Step 1:** Create a reference bone image. The reference bone is a true
representative average of all bone images. This requires segmenting the
trabecular bone from each condyle and making two binary image masks: (1)
a complete bone mask and (2) a trabecular bone mask. Then, using the
complete bone masks for each individual, a random individual is selected
as the initial “reference” image to which each individual is similarity
registered; permitting translation, rotation, and isotropic scaling of
each individual image to the reference image. The masks are then
averaged by taking the arithmetic mean of the similarity translations
and isotropic scaling and taking the quanterion average of the rotation
matrix. The resultant transformation is applied to the reference image
to create *a posteriori* reference image.

**Step 2:** Create a canonical bone image. This canonical bone is a
statistically representative average of all bone images, and is far more
representative than taking a simple average of all bone images. This is
defined by Bezier spline (B-splines) transformations, by first
similarity registering every image to the new reference image to ensure
overlap between the images and then applying a B-Spline transformation
following the parameters used in Bachmann et al. (2022). However, unlike
Bachmann et al. (2022), two additional changes were made. First, a
Limited-memory Broyden–Fletcher–Goldfarb–Shanno 2 (L-BFGS-2) algorithm
(Di Pillo & Murli, 2003) with the following parameterization:
solutionAccuracy=1e-4, numberOfIterations=150, and
deltaConvergenceTolerance=0.001) is applied as it can handle
unconstrained deformations necessary for variable morphological shapes
like in the mandibular condyle. While Bachmann et al. (2022)’s
application of the L-BFGS-B algorithm does allow for constraints of
deformations, it is largely ineffective for very variable morphologies
and can only be rectified with fine parameterization and heavy
computation.

Instead of applying the complete bone mask for the B-Spline
registration, the complete bone mask can be completely filled using a
binary closing algorithm from the scipy package in python 3.9. Then, the
filled binary mask was used for warping. However, the complete bone
image was only used when applying the B-Spline transformation. This
results in an image which matches the reference with slight deviations
in the trabecular bone. This filled bone was used to prevent trabecular
bone deformation collapse which is common when handling very variable
and small trabecular morphologies and sizes such as in the trabecular
bone of the mandibular condyle. To assess registration quality, Dice,
Hausdorff, and Mean Squared Distances (MSD) were used to identify
quantitatively how well the B-Spline operation performed. While the
B-Spline does supply a negative cross-correlation metric, as per
Bachmann et al. (2022), these additional metrics allow quantitative
assessments on how well the registration performed.

Once all of the bones in the study sample are transformed to the
reference, the B-Spline transforms are averaged by converting them to
displacement fields and inverting the averaged displacement field. This
transform was applied to the reference image directly to obtain a
canonical image. This process of similarity registering then B-Spline
registering each image and subsequent canonical bone creation was
iterated several times until the canonical image converged with itself.
For the first iteration, however, since there is no true canonical
image, the average of all metrics across the study sample are used.

**Step 3:** Create a canonical mesh of the canonical bone image. This
step involves applying the B-Spline transformations to the trabecular
bone images rather than the complete bone structure. The transformations
from the first two steps are applied on the individual trabecular bone
masks. After registration, all of the images are averaged into a
canonical trabecular image. This image becomes the blueprint for the
canonical mesh. The canonically averaged trabecular bone image are then
transformed into a solid tetrahedral mesh using the tetgen 0.6.5 package
in Python 3.9. A tetrahedral mesh of the canonical trabecular image uses
a maximum edge length of 1mm by default following Bachmann et
al. (2022). The resulting mesh vertices and tetrahedra ultimately depend
on the complexity and size of the object, and can directly affect its
associated tetrahedral collapse parameter and volume skew.

**Step 4:** Create isotopological meshes for each individual.
Isotopological meshes, or meshes with identical number of corresponding
vertices, are created by applying a reversed B-Spline transformation on
the trabecular canonical mesh obtained from the transformation of the
original trabecular bone image stacks to the canonical bone image. These
isotopological meshes are then used to map the bone volume and
anisotropy from the original trabecular images by applying an
isotopological Holistic Morphometric Analysis of the canonical mesh.
Isotopological Holistic Morphometric Analysis (isoHMA) is designed to
qualitatively analyze morphometric quantities of whole bone structures
without the need of landmark-based approaches or subregions of interest
(Bachmann et al., 2022; Gross, Kivell, Skinner, Nguyen, & Pahr, 2014;
Steiner et al., 2021). While there have been various HMA approaches
(Bachmann et al., 2022), the general workflow has been to standardize
each trabecular image domain, create a volumetric mesh of the trabecular
bone, calculate the morphometric quantities from the bone image, and
interpolate these morphometric values on the generated mesh.
Morphometric values are sampled onto the mesh by creating the same 3D
background grid for each trabecular bone image in a sample. By using
overlapping sampling spheres, the morphometric values of the trabecular
bone are assigned to the 3D background grid and then interpolated onto
the overlapping mesh. This package can calculate two morphometrics
variables commonly used in trabecular studies: (1) bone volume to total
volume (BV/TV) and (2) degree of anisotropy (DA). To standardize these
values more comparable and provide a greater taxonomic signal, each
morphometric scalar is scaled by dividing the value by its mean value
across the trabecular volume to get a relative BV/TV and DA scalar
values.

## Bibliography:

Bachmann, S., Dunmore, C. J., Skinner, M. M., Pahr, D. H., & Synek, A.
(2022). A computational framework for canonical holistic morphometric
analysis of trabecular bone. Scientific Reports, 12(1), 5187.

Bird, E. E., Kivell, T. L., Dunmore, C. J., Tocheri, M. W., & Skinner,
M. M. (2024). Trabecular bone structure of the proximal capitate in
extant hominids and fossil hominins with implications for midcarpal
joint loading and the dart‐thrower’s motion. American Journal of
Biological Anthropology, 183(3), e24824.

Di Pillo, G., & Murli, A. (2003). Quasi-Newton Algorithms for
Large-Scale. High Performance Algorithms Software for Nonlinear
Optimization, 82, 1.

Gross, T., Kivell, T. L., Skinner, M. M., Nguyen, H., & Pahr, D. H.
(2014). A CT-image-based framework for the holistic analysis of cortical
and trabecular bone morphology. Palaeontologia Electronica.

Steiner, L., Synek, A., & Pahr, D. H. (2021). Femoral strength can be
predicted from 2D projections using a 3D statistical deformation and
texture model with finite element analysis. Medical Engineering Physics,
93, 72-82. Summerfield, M. (2014). Python 3: Computer press.

## Python Package:

The cHMA package was made exclusively for the Python 3.9 programming
language. This package consists of four primary functions to carry out
an entire cHMA analysis as described above:

1.  **resample:** Resample all 3D images to the same dimensions, size,
    and spacing.

2.  **cHMA:** Create the canonical bone image.

3.  **isoHMA:** Conduct Holistic Morphometric Analysis by generating a
    canonical and isotopological bone meshes.

4.  **smesh:** Assign scalar values to the canonical mesh.

## resample:

``` python
def resample_condyles(input_dir, output_dir, expected_spacing, cores='detect'):
    """
    Dynamically resample all images to the same dimensions and size.
    
    Parameters:
    -----------
    input_dir : str
        Path to input mesh file (.vtk, .vol, etc.)
    pc_loadings : array-like
        PC loadings (one per vertex)
    output_file : str
        Path to output VTK file
    pc_name : str
        Name of the PC component (default: "PC1") or scalar value name
        
    Returns:
    --------
    str
        Path to the created output file
    """
    """
```

## cHMA:

``` python
def cHMA(input_dir, output_dir, reference_name="reference", scale_factor=3, max_iterations=5, cp=[1,1,1], cores='detect'):
    """
    Perform Canonical Holistic Morphometric Analysis (cHMA) on trabecular bone images.
    
    Parameters:
    -----------
    input_dir : (str)
        Directory containing filled and trabecular/cortical binary images
    output_dir : (str)
        Directory to save output files from the cHMA analysis 
    reference_name : (str)
        Name of the reference condyle (default: "reference")
    scale_factor : (int)
        Factor by which to reduce image resolution for faster processing (default: 3)
    max_iterations : (int)
        Maximum number of iterations for canonical bone creation (default: 5)
    cp : (list)
        Control point grid spacing in mm for B-spline registration [x, y, z] (default: [1, 1, 1])
    cores : (int)
        How many cores should be used. Default is max number of cores - 3.
    
    Returns:
    --------
    bool: True if successful, False otherwise
    Resulting files are stored in the output directory
    """
```

## isoHMA:

``` python
def isohma(input_dir, output_dir, iteration=2, cores = 'detect', reference="reference", method="chma"):
    """
    Create a canonical and corresponding isotopological meshes through the Holistic Morphometric Analysis procedure. 
    
    Parameters:
    -----------
    input_dir : str
        The output directory of the cHMA function.
    output_dir : str
        Directory to save output files. This should ideally be the same as the input directory.
    iteration : int
        Number of the last iteration from the canonical Holistic Morphometric Analysis desired for analysis.
        
    Returns:
    --------
    bool: True if successful, False otherwise
    Resulting files are stored in the output directory
    """
    
```

## smesh:

``` python
def scalar_mesh(input_file, pc_loadings, output_file, pc_name="PC1"):
    """
    Add scalar values or PC loadings to an existing mesh file (.vtk, .vol).
    
    This function supports the cHMA methodology described in Bachmann et al. 2022
    for trabecular bone analysis by adding principal component loadings as scalar
    values to mesh vertices.
    
    Parameters:
    -----------
    input_file : str
        Path to input mesh file (.vtk, .vol, etc.)
    pc_loadings : array-like
        PC loadings (one per vertex)
    output_file : str
        Path to output VTK file
    pc_name : str
        Name of the PC component (default: "PC1") or scalar value name
        
    Returns:
    --------
    str
        Path to the created output file
    """
```

## Example: **Canonical Holistic Morphometric Analysis (cHMA) of binary image masks of Right Mandibular Condyles**

------------------------------------------------------------------------

## **Step 1: Resample all images to same dimensions and size**

``` python
# Right Side Resampling
import cHMA
import gc

input_dir = "B:/Test"
output_dir = "B:/Test"
expected_spacing = [0.06,0.06,0.06]

cHMA.resample(input_dir, output_dir, expected_spacing, cores=28)
```

## **Step 2: Create the Canonical Bone with function cHMA**

``` python
import cHMA 
import gc

gc.collect()

input_dir = "B:/Test"
output_dir = "B:/Test"

cHMA.cHMA(input_dir, output_dir, reference_name="1", scale_factor=5, max_iterations=2, cp=[1.5,1.5,1.5], cores=28)

gc.collect()
```

## **Step 3: Conduct the HMA analysis using the canonical trabecular bone image**

``` python
import cHMA
import gc

gc.collect()

input_dir = "B:/Test"
output_dir = "B:/Test"
cHMA.isoHMA(input_dir, output_dir, iteration=2, cores = 28, reference="1278", method="chma")
```

## **Step 4: Map Principal Component Extremes from Python (after analysis) to the Canonical Mesh**

``` python
import cHMA
import pandas as pd
import numpy as np
import os

# Data lists
data = [
    "PC_neg1", "PC_pos1", "PC_neg2", "PC_pos2"
]

# Loop through data and cb together using zip
for pc_name, cb_folder in zip(data, cb):
    cHMA.smesh(
        input_file=f"B:/Test/Canonical_Bone/trabecular.vtk",
        pc_loadings=pd.read_csv(f"B:/CResults/{pc_name}.csv")["PC_Loading"].values,
        output_file=f"B:/Test/{pc_name}.vtk",
        pc_name=pc_name
    )
```

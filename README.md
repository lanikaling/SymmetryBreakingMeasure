# A Continuous Symmetry Breaking Measure for Finite Clusters

This repository contains the implementation of the method described in the paper, which will be published soon. If you use this repository, please make sure to cite the paper once it is available.

## Description

This repository implements a continuous symmetry-breaking measure for finite atomic clusters using Kullback-Leibler (KL) and Jensen-Shannon (JS) divergence.

In this model, a finite cluster of atoms is represented as a normalized electron-weighted atomic density function, where the electron density at each atom is modeled as a Gaussian distribution centered at the atomic nucleus. To measure symmetry breaking, we apply KL and JS divergence between the original density function and the transformed function after applying a symmetry operation. The resulting values provide a quantitative measure of symmetry breaking with respect to that operation.

Unlike traditional methods that give binary results, this approach offers a detailed, quantitative assessment of how much a structure deviates from a given symmetry transformation. It is a valuable tool for researchers studying symmetry-breaking distortions in finite atomic clusters, providing insights into how these distortions influence material properties.

## Installation

This repository requires **DiffPy-CMI** for structure modeling and analysis. You can find the installation instructions on the [official DiffPy-CMI website](https://www.diffpy.org/products/diffpycmi/install.html).

## Usage

To calculate the symmetry-breaking measure (SBM) of a finite cluster, you need to first define a `FiniteCluster` class object.

### Defining a FiniteCluster Object

The finite cluster information is stored in `atoms_info`, a `pd.DataFrame` that contains the following columns:

- `"x"`, `"y"`, `"z"`: Cartesian coordinates of atoms.
- `"num_electrons"`: Number of electrons in each atom.
- `"uiso"`: Squared standard deviation of the atom's Gaussian distribution (Uiso).
- `"occupancy"`: Fractional occupancy of each atom (default is 1).

You can create the `FiniteCluster` object by manually defining a `pd.DataFrame` following the format above, or by using crystallographic data.

#### Importing from CIF File
If you are working with crystals and want to use the unit cell atoms as the finite cluster, use the `import_unit_cell_from_cif` method:

```
finite_cluster.import_unit_cell_from_cif(cif_directory)
```

This will automatically import the unit cell and atomic structure information from a CIF (Crystallographic Information File).

If you want to apply defects or local distortions, you can directly modify the `atoms_info` of your `FiniteCluster` class object. Additionally, if you want to define a finite cluster for a supercell, you can use the method:

```
finite_cluster.expand_by_supercell(num_supercells)
```

### Calculating Symmetry-Breaking Measure (SBM)

To calculate the SBM of a finite cluster with respect to a specific symmetry operator, use the following steps:

1. **Calculate Suggested Monte Carlo (MC) Sample Size**:

```
sample_size, estimated_sbm = finite_cluster.calc_symmetry_breaking_measure_sample_size(
    num_samples=NUM_SAMPLES,
    operator=operator,
    confidence_interval=CONFIDENCE_LEVEL,
    tolerance_single_side=TOLERANCE_ONE_SIDE,
    method=method,
    **operator_kwrgs
)
```

This calculates the suggested MC sample size based on the `CONFIDENCE_LEVEL` and `TOLERANCE_ONE_SIDE`. If you already have a sample size, you can skip this step.

2. **Calculate the Symmetry-Breaking Measure**:

```
measure = finite_cluster.calc_symmetry_breaking_measure(
    num_samples=sample_size,
    operator=operator,
    method=method,
    **operator_kwrgs
)
```

#### Symmetry Operators
Currently supported symmetry operators include:
- Reflection
- Rotation
- Translation
- Inversion

#### Methods
- **KL**: Kullback-Leibler divergence.
- **JS**: Jensen-Shannon divergence.

## API Documentation

### `FiniteCluster`

The `FiniteCluster` class handles the representation and manipulation of a finite atomic cluster.

- `import_unit_cell_from_cif(cif_directory: str)`: Imports atomic structure information from a CIF file and initializes the cluster.
- `expand_by_supercell(num_supercells: int)`: Expands the current unit cell to define a finite cluster from a supercell.

### Symmetry-Breaking Measure Methods

- `calc_symmetry_breaking_measure_sample_size(...)`: Calculates the recommended Monte Carlo sample size for the SBM calculation.
- `calc_symmetry_breaking_measure(...)`: Computes the symmetry-breaking measure using KL or JS divergence.

For more detailed documentation of each method, refer to the codebase.

## Citation

If you use this repository in your research, please cite the following paper:

**Title**: A continuous symmetry breaking measure for finite clusters using Jensen-Shannon divergence
**Authors**: Ling Lan, Qiang Du, Simon J. L. Billinge
**Journal**: [To be filled in]
**DOI**: [To be filled in]

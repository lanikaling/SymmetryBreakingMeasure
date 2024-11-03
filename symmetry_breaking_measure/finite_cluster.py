from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from diffpy.structure import Lattice, loadStructure
from scipy.stats import norm

from symmetry_breaking_measure.base_operator import BaseOperator
from symmetry_breaking_measure.constants import ATOMIC_NUMBER


class FiniteCluster:
    """
    Represents a finite atomic cluster with optional lattice parameters.
    This class allows calculation of symmetry-breaking measures with respect to
    local symmetry operations (rotation, reflection, translation, inversion).

    Parameters
    ----------
    atoms_info : pd.DataFrame, optional
        A DataFrame containing atomic information with the following columns:
        - "x", "y", "z": Cartesian coordinates of atoms.
        - "num_electrons": Number of electrons in each atom.
        - "uiso": Squared standard deviation of the atom's Gaussian distribution (Uiso).
        - "occupancy": Fractional occupancy of each atom, default is 1.
    lattice : Lattice, optional
        Lattice parameters of the unit cell, if applicable.

    Attributes
    ----------
    atoms_info : pd.DataFrame
        DataFrame containing atomic site information.
    lattice : Lattice or None
        The lattice object defining the unit cell, or None if not specified.
    """

    def __init__(
        self,
        atoms_info: pd.DataFrame = pd.DataFrame(
            columns=["x", "y", "z", "num_electrons", "uiso", "occupancy"]
        ),
        lattice: Lattice = None,
    ):
        self._atoms_info = atoms_info
        self._lattice = lattice

    @property
    def atoms_info(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            A DataFrame containing atomic site information including
            Cartesian coordinates, number of electrons, Uiso, and occupancy.
        """
        return self._atoms_info

    @property
    def lattice(self) -> Lattice:
        """
        Returns
        -------
        Lattice or None
            The lattice parameters of the unit cell, or None if no unit cell is specified.
        """
        return self._lattice

    @property
    def xyz(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            A NumPy array containing the Cartesian coordinates (x, y, z) of all atoms in the cluster.
        """
        return self._atoms_info[["x", "y", "z"]].to_numpy()

    def set_lattice(self, new_lattice: Lattice):
        """
        Sets the lattice parameters for the unit cell.

        Parameters
        ----------
        new_lattice : Lattice
            The new lattice parameters to assign to the unit cell.
        """
        self._lattice = new_lattice

    def set_xyz(self, new_xyz: np.ndarray):
        """
        Sets the Cartesian coordinates (x, y, z) for the atoms in the finite cluster.

        Parameters
        ----------
        new_xyz : np.ndarray
            A NumPy array containing the Cartesian coordinates of the atoms in the cluster.
            This array should have at least three columns (x, y, z) and the number of rows
            must match the number of atoms in the current `atoms_info`.

        Behavior
        --------
        - If the number of electrons for any atom is not specified (NaN), it defaults to 1.
        - If the Uiso for any atom is not specified (NaN), it defaults to 1/(8π²).
        - If the occupancy for any atom is not specified (NaN), it defaults to 1.

        Raises
        ------
        ValueError
            If the number of rows in `new_xyz` does not match the number of atoms
            already present in `atoms_info`.
        """
        # Check if the number of rows in new_xyz matches the current number of rows in self._atoms_info
        current_num_of_atoms = len(self._atoms_info.index)
        if current_num_of_atoms > 0 and new_xyz.shape[0] != current_num_of_atoms:
            raise ValueError(
                f"new_xyz should have {current_num_of_atoms} rows to match the current data, but it has {new_xyz.shape[0]} rows."
            )

        self._atoms_info[["x", "y", "z"]] = new_xyz
        num_of_atoms = new_xyz.shape[0]

        if self._atoms_info["num_electrons"].isnull().all():
            self._atoms_info["num_electrons"] = np.ones((num_of_atoms, 1))
        if self._atoms_info["uiso"].isnull().all():
            self._atoms_info["uiso"] = np.ones((num_of_atoms, 1)) * 1 / (8 * np.pi**2)
        if self._atoms_info["occupancy"].isnull().all():
            self._atoms_info["occupancy"] = np.ones((num_of_atoms, 1))

    def set_num_electrons(self, new_num_electrons: Union[int, float, np.ndarray]):
        """
        Sets the number of electrons for atoms in the cluster.

        Parameters
        ----------
        new_num_electrons : int, float, or np.ndarray
            The new number of electrons for the atoms. This can be:
            - A single integer or float, which will be applied to all atoms.
            - A NumPy array where each entry corresponds to the number of electrons
            for each atom. The length of the array must match the number of atoms
            in the current `atoms_info`.

        Raises
        ------
        ValueError
            If `new_num_electrons` is provided as a NumPy array and its length
            does not match the number of atoms in `atoms_info`.
        """
        if isinstance(new_num_electrons, (float, int)):
            self._atoms_info["num_electrons"] = new_num_electrons
        else:
            if len(self._atoms_info.index) != new_num_electrons.shape[0]:
                raise ValueError(
                    "The number of rows in new_num_electrons must match the number of atoms."
                )
            self._atoms_info["num_electrons"] = new_num_electrons

    def set_uiso(self, new_uiso: Union[int, float, np.ndarray]):
        """
        Sets the isotropic atomic displacement parameter (Uiso) for atoms in the cluster.

        Parameters
        ----------
        new_uiso : int, float, or np.ndarray
            The new Uiso values to assign. This can be:
            - A single integer or float, which will be applied to all atoms.
            - A NumPy array where each entry specifies the Uiso value for each atom.
            The length of the array must match the number of atoms in the current `atoms_info`.

        Raises
        ------
        ValueError
            If `new_uiso` is provided as a NumPy array and its length does not match
            the number of atoms in `atoms_info`.
        """
        if isinstance(new_uiso, (float, int)):
            self._atoms_info["uiso"] = new_uiso
        else:
            if len(self._atoms_info.index) != new_uiso.shape[0]:
                raise ValueError(
                    "The number of rows in new_uiso must match the number of atoms."
                )
            self._atoms_info["uiso"] = new_uiso

    def set_occupancy(self, new_occupancy: Union[int, float, np.ndarray]):
        """
        Sets the occupancy for atoms in the cluster.

        Parameters
        ----------
        new_occupancy : int, float, or np.ndarray
            The new occupancy values to assign. This can be:
            - A single integer or float, which will be applied to all atoms.
            - A NumPy array where each entry specifies the occupancy for each atom.
            The length of the array must match the number of atoms in the current `atoms_info`.

        Raises
        ------
        ValueError
            If `new_occupancy` is provided as a NumPy array and its length does not match
            the number of atoms in `atoms_info`.
        """
        if isinstance(new_occupancy, (float, int)):
            self._atoms_info["occupancy"] = new_occupancy
        else:
            if len(self._atoms_info.index) != new_occupancy.shape[0]:
                raise ValueError(
                    "The number of rows in new_occupancy must match the number of atoms."
                )
            self._atoms_info["occupancy"] = new_occupancy

    def import_unit_cell_from_cif(self, cif_directory: str) -> None:
        """
        Imports unit cell and atomic structure information from a CIF (Crystallographic Information File).

        Parameters
        ----------
        cif_directory : str
            The file path to the CIF file containing the unit cell structure information.

        Behavior
        --------
        - Parses the CIF file located at the provided directory.
        - Loads the lattice parameters and sets the `lattice` attribute.
        - Imports the atomic Cartesian coordinates and sets the `xyz` attribute using `set_xyz`.
        - Calculates and sets the number of electrons for each atom using `set_num_electrons`.
        - Sets the isotropic atomic displacement parameters (Uiso) using `set_uiso`.
        - Sets the atomic occupancy values using `set_occupancy`.

        Raises
        ------
        FileNotFoundError
            If the provided CIF file cannot be found at the specified directory.
        ValueError
            If the CIF file does not contain valid structure data or required fields.
        """

        # Parsing the cif file
        structure = loadStructure(cif_directory)
        self._lattice = structure.lattice
        self.set_xyz(structure.xyz_cartn)
        self.set_num_electrons(self.calc_num_electrons(list(structure.element)))
        self.set_uiso(structure.Uisoequiv)
        self.set_occupancy(structure.occupancy)

    @staticmethod
    def calc_num_electrons(symbol_list: List[str]) -> np.ndarray:
        """
        Calculates the number of electrons for each atom or ion in the given list of species.

        Parameters
        ----------
        symbol_list : List[str]
            A list of strings, where each string represents an atom or ion species.
            The species can optionally include a charge, such as "Ca2+" or "Cl-".
            If no charge is specified (e.g., "H"), the atom is considered neutral.

        Returns
        -------
        num_of_electrons : np.ndarray
            A NumPy array where each element represents the number of electrons for
            the corresponding atom or ion in `symbol_list`.
        """
        num_electrons = np.zeros(len(symbol_list))
        for i, symbol in enumerate(symbol_list):
            if symbol[-1] not in ["+", "-"]:
                num_electrons[i] = ATOMIC_NUMBER[symbol]
            else:
                species = "".join([s for s in symbol if s.isalpha()])
                species_val = ATOMIC_NUMBER[species]
                charges = "".join([s for s in symbol if s.isdigit()])
                charges = float(charges) if charges else 1
                sign = 1 if symbol[-1] == "+" else -1
                num_electrons[i] = species_val - sign * charges
        return num_electrons

    def frac_to_cart(self, frac_xyz: np.ndarray) -> np.ndarray:
        """
        Converts atom positions from fractional coordinates to Cartesian coordinates.

        Parameters
        ----------
        frac_xyz : np.ndarray
            A NumPy array with at least three columns, where each row represents the
            fractional (x, y, z) coordinates of an atom in the cluster.

        Returns
        -------
        cart_xyz : np.ndarray
            A NumPy array where the first three columns represent the Cartesian
            coordinates of the atoms, converted from the given fractional coordinates.
        """
        cart_xyz = np.matmul(frac_xyz, self._lattice.stdbase)
        return cart_xyz

    def cart_to_frac(self, cart_xyz: np.ndarray) -> np.ndarray:
        """
        Converts atom positions from Cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        cart_xyz : np.ndarray
            A NumPy array with at least three columns, where each row represents the
            Cartesian (x, y, z) coordinates of an atom in the cluster.

        Returns
        -------
        frac_xyz : np.ndarray
            A NumPy array where the first three columns represent the fractional
            coordinates of the atoms, converted from the given Cartesian coordinates.
        """
        frac_xyz = np.matmul(cart_xyz, np.linalg.inv(self._lattice.stdbase))
        return frac_xyz

    def generate_samples(
        self,
        num_samples: int,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
    ) -> np.ndarray:
        """
        Generates random samples based on the electron density of the structure
        specified by `atoms_info`. Each atom is modeled as a 3D spherical Gaussian
        distribution with a variance determined by the Uiso value of the atom.

        Parameters
        ----------
        num_samples : int
            The number of random samples to generate.
        atoms_info : pd.DataFrame, optional
            A DataFrame containing atomic information (coordinates, electron numbers,
            Uiso, and occupancy). If not provided, the default is to use the
            instance's `self._atoms_info`.
        random_seed : int, optional
            The random seed for the NumPy random number generator. Defaults to 0.

        Returns
        -------
        samples : np.ndarray
            A NumPy array of shape (num_samples, 3), where each row is a random
            sample representing a position in 3D space. The samples are generated
            from a Gaussian mixture model based on the atoms' electron density and
            other properties from `atoms_info`.
        """
        np.random.seed(random_seed)
        if atoms_info is None:
            atoms_info = self._atoms_info

        phi = np.multiply(atoms_info["num_electrons"], atoms_info["occupancy"])
        phi_sum = np.cumsum(phi)
        phi_sum = phi_sum / phi_sum.iloc[-1]

        # Generate random numbers outside the loop
        u = np.random.uniform(size=num_samples)
        k_indices = np.searchsorted(phi_sum, u)

        # Batch generate samples using vectorized operations
        sigma = np.sqrt(atoms_info.loc[k_indices, "uiso"])
        x_samples = np.random.normal(atoms_info.loc[k_indices, "x"], sigma)
        y_samples = np.random.normal(atoms_info.loc[k_indices, "y"], sigma)
        z_samples = np.random.normal(atoms_info.loc[k_indices, "z"], sigma)

        # Combine samples into a single array
        samples = np.column_stack((x_samples, y_samples, z_samples))
        return samples

    def _calc_atoms_info_transformed(
        self,
        operator: BaseOperator,
        atoms_info: pd.DataFrame = None,
        **operator_kwrgs,
    ) -> pd.DataFrame:
        """
        Applies a transformation to the atomic coordinates using the specified operator.

        Parameters
        ----------
        operator : BaseOperator
            An operator object that defines the transformation to apply to the atomic
            coordinates.
        atoms_info : pd.DataFrame, optional
            A DataFrame containing atomic information, including the coordinates ("x", "y", "z").
            If not provided, the instance's `self._atoms_info` is used.
        **operator_kwrgs
            Additional keyword arguments passed to the operator's `apply` method.

        Returns
        -------
        atoms_info_transformed : pd.DataFrame
            A DataFrame containing the transformed atomic coordinates, with the same structure
            as the input `atoms_info`.
        """
        if atoms_info is None:
            atoms_info = self._atoms_info
        atoms_info_transformed = atoms_info.copy()
        atoms_info_transformed[["x", "y", "z"]] = operator.apply(
            atoms_xyz=atoms_info[["x", "y", "z"]].to_numpy(), **operator_kwrgs
        )
        return atoms_info_transformed

    def _calc_atoms_info_averaged(
        self,
        operator: BaseOperator,
        atoms_info: pd.DataFrame = None,
        **operator_kwrgs,
    ) -> [pd.DataFrame, pd.DataFrame]:
        """
        Applies a transformation to the atomic coordinates and computes the averaged
        atomic information by combining the original and transformed data.

        Parameters
        ----------
        operator : BaseOperator
            An operator object that defines the transformation to apply to the atomic
            coordinates.
        atoms_info : pd.DataFrame, optional
            A DataFrame containing atomic information, including the coordinates ("x", "y", "z").
            If not provided, the instance's `self._atoms_info` is used.
        **operator_kwrgs
            Additional keyword arguments passed to the operator's `apply` method.

        Returns
        -------
        atoms_info_transformed : pd.DataFrame
            A DataFrame containing the transformed atomic coordinates.
        atoms_info_averaged : pd.DataFrame
            A DataFrame containing the combined original and transformed atomic
            information.
        """
        if atoms_info is None:
            atoms_info = self._atoms_info
        atoms_info_transformed = self._calc_atoms_info_transformed(
            operator=operator,
            atoms_info=atoms_info,
            **operator_kwrgs,
        )
        atoms_info_averaged = atoms_info.copy()
        atoms_info_averaged = atoms_info_averaged.append(
            atoms_info_transformed, ignore_index=True
        )  # Ignoring index to avoid potential duplicate index issues
        atoms_info_averaged["occupancy"] = atoms_info_averaged["occupancy"] / 2
        return atoms_info_transformed, atoms_info_averaged

    def calc_symmetry_breaking_measure(
        self,
        num_samples: Union[int, list],
        operator: BaseOperator,
        method: str,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
        **operator_kwrgs,
    ) -> float:
        """
        Calculates the symmetry-breaking measure using the specified method and transformation.

        Parameters
        ----------
        num_samples : int or list
            The number of random samples to generate for the calculation. For the "KL" method,
            this should be an integer. For the "JS" method, this should be a list containing two
            integers, corresponding to the sample sizes for each distribution.
        operator : BaseOperator
            The operator object that defines the transformation to apply to the atomic
            coordinates.
        method : str
            The method used to calculate the symmetry-breaking measure. Supported methods are:
            - "KL" : Kullback-Leibler divergence
            - "JS" : Jensen-Shannon divergence
        atoms_info : pd.DataFrame, optional
            A DataFrame containing atomic information, including the coordinates ("x", "y", "z").
            If not provided, the instance's `self._atoms_info` is used.
        random_seed : int, optional
            The random seed for the NumPy random number generator. Defaults to 0.
        **operator_kwrgs
            Additional keyword arguments passed to the operator's `apply` method.

        Returns
        -------
        float
            The calculated symmetry-breaking measure, based on the specified method.

        Behavior
        --------
        - "KL" (Kullback-Leibler divergence): Compares the original atomic distribution with
        the transformed one.
        - "JS" (Jensen-Shannon divergence): A symmetrized version of KL divergence that averages
        the KL divergences between the original, transformed, and averaged distributions.
        """
        if atoms_info is None:
            atoms_info = self._atoms_info

        if method == "KL":
            atoms_info_transformed = self._calc_atoms_info_transformed(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )
            measure = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_transformed,
                atoms_info=atoms_info,
                random_seed=random_seed,
            )
            return measure

        elif method == "JS":
            (
                atoms_info_transformed,
                atoms_info_averaged,
            ) = self._calc_atoms_info_averaged(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )

            measure_1 = self.calc_kl_divergence(
                num_samples=num_samples[0],
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info,
                random_seed=random_seed,
            )

            measure_2 = self.calc_kl_divergence(
                num_samples=num_samples[1],
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info_transformed,
                random_seed=random_seed,
            )

            return 0.5 * measure_1 + 0.5 * measure_2

    def calc_symmetry_breaking_measure_sample_size(
        self,
        num_samples: int,
        operator: BaseOperator,
        confidence_interval,
        tolerance_single_side,
        method: str,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
        **operator_kwrgs,
    ) -> float:
        """
        Calculates the sample size required to estimate the symmetry-breaking measure
        within a given confidence interval and tolerance using the specified method.

        Parameters
        ----------
        num_samples : int
            The number of initial samples to generate for the calculation.
        operator : BaseOperator
            The operator object that defines the transformation to apply to the atomic
            coordinates.
        confidence_interval : float
            The desired confidence interval for the estimation of the symmetry-breaking measure.
        tolerance_single_side : float
            The tolerance value on one side of the confidence interval.
        method : str
            The method used to calculate the symmetry-breaking measure. Supported methods are:
            - "KL" : Kullback-Leibler divergence
            - "JS" : Jensen-Shannon divergence
        atoms_info : pd.DataFrame, optional
            A DataFrame containing atomic information, including the coordinates ("x", "y", "z").
            If not provided, the instance's `self._atoms_info` is used.
        random_seed : int, optional
            The random seed for the NumPy random number generator. Defaults to 0.
        **operator_kwrgs
            Additional keyword arguments passed to the operator's `apply` method.

        Returns
        -------
        sample_size : int or list of int
            The calculated sample size required to achieve the specified confidence interval
            and tolerance. For "KL", a single integer is returned. For "JS", a list containing
            the sample sizes for the two distributions and the overall sample size is returned.
        measure : float
            The calculated symmetry-breaking measure based on the specified method.

        Behavior
        --------
        - For "KL", the function calculates the Kullback-Leibler divergence between the original
        and transformed atomic distributions and determines the required sample size.
        - For "JS", the function calculates the Jensen-Shannon divergence and determines the
        sample size for both distributions (original vs. averaged, transformed vs. averaged)
        and the overall sample size.
        """
        if atoms_info is None:
            atoms_info = self._atoms_info

        if method == "KL":
            atoms_info_transformed = self._calc_atoms_info_transformed(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )

            measure, sample_values = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_transformed,
                atoms_info=atoms_info,
                random_seed=random_seed,
                return_sample_values=True,
            )

            sample_size = self.calc_sample_size(
                sample_values=sample_values,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )
            return sample_size, measure

        if method == "JS":
            (
                atoms_info_transformed,
                atoms_info_averaged,
            ) = self._calc_atoms_info_averaged(
                operator=operator,
                atoms_info=atoms_info,
                **operator_kwrgs,
            )

            measure1, sample_values1 = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info,
                random_seed=random_seed,
                return_sample_values=True,
            )
            sample_size1 = self.calc_sample_size(
                sample_values=sample_values1,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )

            measure2, sample_values2 = self.calc_kl_divergence(
                num_samples=num_samples,
                atoms_info_transformed=atoms_info_averaged,
                atoms_info=atoms_info_transformed,
                random_seed=random_seed,
                return_sample_values=True,
            )
            sample_size2 = self.calc_sample_size(
                sample_values=sample_values2,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )
            sample_size_overall = self.calc_sample_size(
                sample_values=sample_values1 + sample_values2,
                confidence_interval=confidence_interval,
                tolerance_single_side=tolerance_single_side,
            )
            sample_size = [sample_size1, sample_size2, sample_size_overall]
            measure = 0.5 * measure1 + 0.5 * measure2
            return sample_size, measure

    def calc_sample_size(
        self,
        sample_values: np.ndarray,
        confidence_interval,
        tolerance_single_side,
    ) -> int:
        """
        Calculates the required sample size to estimate a parameter within a given
        confidence interval and tolerance.

        Parameters
        ----------
        sample_values : np.ndarray
            A NumPy array containing the sample values used for the calculation.
        confidence_interval : float
            The desired confidence interval for the estimation (e.g., 0.95 for a 95% confidence level).
        tolerance_single_side : float
            The tolerance limit for the estimate on one side of the confidence interval.

        Returns
        -------
        sample_size : int
            The calculated sample size required to achieve the specified confidence interval
            and tolerance.

        Behavior
        --------
        - The method calculates the z-value corresponding to the given confidence level using
        the normal distribution.
        - The standard deviation of the sample values is used to estimate the variability in
        the data.
        - The sample size is determined by the formula:
        (z_value * st_dev / tolerance_single_side)², and it is rounded up to the next integer.
        """
        alpha = 1 - confidence_interval
        z_value = norm.ppf(
            1 - alpha / 2
        )  # calculates the z-score at the (1-alpha/2) percentile
        st_dev = np.sqrt(np.var(sample_values, axis=0))
        sample_size = np.ceil(((st_dev * z_value) / tolerance_single_side) ** 2)
        return int(sample_size)

    def calc_kl_divergence(
        self,
        num_samples: int,
        atoms_info_transformed: pd.DataFrame,
        atoms_info: pd.DataFrame = None,
        random_seed: int = 0,
        return_sample_values: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Calculates the Kullback-Leibler (KL) divergence as a symmetry-breaking measure
        between the original structure and a transformed structure using Monte Carlo sampling.

        Parameters
        ----------
        num_samples : int
            The number of random samples to generate for the calculation.
        atoms_info_transformed : pd.DataFrame
            A DataFrame containing the transformed atomic structure information,
            including coordinates and other relevant properties.
        atoms_info : pd.DataFrame, optional
            A DataFrame containing the original atomic structure information. If not provided,
            the instance's `self._atoms_info` is used as the reference structure.
        random_seed : int, optional
            The random seed for the NumPy random number generator. Defaults to 0.
        return_sample_values : bool, optional
            If True, returns the sample values used in the calculation along with the KL divergence.

        Returns
        -------
        float
            The calculated KL divergence, representing the symmetry-breaking measure between
            the original structure and the transformed structure.
        Tuple[float, np.ndarray], optional
            If `return_sample_values` is True, returns a tuple containing the KL divergence
            and the array of sample values.

        Behavior
        --------
        - The method generates random samples based on the original structure using a Monte Carlo
        approach.
        - The Gaussian probability density function (PDF) is calculated for both the original and
        transformed structures.
        - KL divergence is computed using the logarithm of the ratio of the PDFs between the original
        and transformed structures, and the result is averaged across samples.
        """
        if atoms_info is None:
            atoms_info = self._atoms_info

        num_atoms, std_values, phi = self._calc_dim_std_phi(atoms_info=atoms_info)
        (
            num_atoms_transformed,
            std_values_transformed,
            phi_transformed,
        ) = self._calc_dim_std_phi(atoms_info=atoms_info_transformed)
        samples = self.generate_samples(
            num_samples=num_samples,
            atoms_info=atoms_info,
            random_seed=random_seed,
        )
        # Use comprehension loop for p and sp calculations
        p = np.array(
            [
                self._calc_gaussian_pdf_3d(
                    x=samples,
                    mean=atoms_info.iloc[i, :3].to_numpy(),
                    cov_scalar=std_values[i].reshape(1, -1, 1),
                )
                for i in range(num_atoms)
            ]
        ).T
        q = np.array(
            [
                self._calc_gaussian_pdf_3d(
                    x=samples,
                    mean=atoms_info_transformed.iloc[i, :3].to_numpy(),
                    cov_scalar=std_values_transformed[i].reshape(1, -1, 1),
                )
                for i in range(num_atoms_transformed)
            ]
        ).T

        P = np.matmul(p, phi).astype(float)
        Q = np.matmul(q, phi_transformed).astype(float)
        sample_values = np.log2(P / Q) + np.log2(sum(phi_transformed) / sum(phi))
        kl_divergence = np.mean(sample_values)

        if return_sample_values:
            return kl_divergence, sample_values

        return kl_divergence

    def _calc_dim_std_phi(
        self,
        atoms_info: pd.DataFrame = None,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Calculates the number of atoms, the standard deviation values (sqrt of Uiso),
        and the weighting factor (phi) for each atom based on atomic information.

        Parameters
        ----------
        atoms_info : pd.DataFrame, optional
            A DataFrame containing atomic information, including the columns "num_electrons",
            "uiso", and "occupancy". If not provided, the instance's `self._atoms_info` is used.

        Returns
        -------
        num_atoms : int
            The total number of atoms in the atomic structure.
        std_values : np.ndarray
            A NumPy array containing the standard deviation (sqrt of Uiso) for each atom.
        phi : np.ndarray
            A NumPy array containing the weighting factors for each atom, calculated as the
            product of the number of electrons and occupancy, divided by Uiso.
        """
        if atoms_info is None:
            atoms_info = self._atoms_info

        num_atoms = atoms_info.shape[0]

        # Convert DataFrame columns to numpy arrays
        num_electrons = atoms_info["num_electrons"].to_numpy()
        uiso = atoms_info["uiso"].to_numpy()
        occupancy = atoms_info["occupancy"].to_numpy()

        phi = np.divide(num_electrons * occupancy, uiso)
        std_values = np.sqrt(uiso)
        return num_atoms, std_values, phi

    @staticmethod
    def _calc_gaussian_pdf_3d(
        x: np.ndarray, mean: np.ndarray, cov_scalar: np.ndarray
    ) -> float:
        """
        Calculates the probability density of a 3D Gaussian distribution at a given point.

        Parameters
        ----------
        x : np.ndarray
            The point where the probability density is calculated. Should be a 3D coordinate.
        mean : np.ndarray
            The mean of the 3D Gaussian distribution, represented as a 3D coordinate.
        cov_scalar : np.ndarray
            A scalar value multiplied by the identity matrix to form the covariance matrix
            for the spherical Gaussian distribution.

        Returns
        -------
        gaussian_pdf_3d : float
            The probability density of the 3D Gaussian distribution at the point `x`.

        Behavior
        --------
        - The method computes the displacement from the mean and applies it to the Gaussian
        probability density function.
        - Assumes the distribution is spherical, so the covariance matrix is scaled by the
        provided `cov_scalar` value.
        """
        disp = x - mean
        power = -0.5 * np.sum(disp**2, axis=-1) / cov_scalar.squeeze(-1)
        denominator = np.sqrt((2 * np.pi * cov_scalar.squeeze()) ** 3)
        gaussian_pdf_3d = 1 / denominator * np.exp(power)

        return gaussian_pdf_3d

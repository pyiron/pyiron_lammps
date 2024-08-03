from pylammpsmpi import LammpsASELibrary


def calculation(funct: callable) -> callable:
    """
    A decorator that wraps a function with LAMMPS instance management.

    Args:
        funct (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function.
    """

    def funct_return(lmp: LammpsASELibrary = None, *args, **kwargs) -> any:
        """
        The wrapped function that manages the LAMMPS instance.

        Args:
            lmp (LammpsASELibrary, optional): The LAMMPS instance. If None, a temporary instance will be created.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            any: The result of the wrapped function.
        """
        # Create temporary LAMMPS instance if necessary
        if lmp is None:
            close_lmp_after_calculation = True
            lmp = LammpsASELibrary()
        else:
            close_lmp_after_calculation = False

        # Run function
        result = funct(lmp=lmp, *args, **kwargs)

        # Close temporary LAMMPS instance
        if close_lmp_after_calculation:
            lmp.close()
        return result

    return funct_return

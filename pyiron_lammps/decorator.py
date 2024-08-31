from pylammpsmpi import LammpsASELibrary


def calculation(funct: callable) -> callable:
    """
    A decorator that wraps a function with LAMMPS instance management.

    Args:
        funct (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function.
    """

    def funct_return(
        lmp: LammpsASELibrary = None, enable_mpi: bool = False, *args, **kwargs
    ) -> any:
        """
        The wrapped function that manages the LAMMPS instance.

        Args:
            lmp (LammpsASELibrary, optional): The LAMMPS instance. If None, a temporary instance will be created.
            enable_mpi (bool): Flag to enable MPI. Default is False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            any: The result of the wrapped function.
        """
        # Create temporary LAMMPS instance if necessary
        if lmp is None:
            close_lmp_after_calculation = True
            if enable_mpi:
                # To get the right instance of MPI.COMM_SELF it is necessary to import it inside the function.
                from mpi4py import MPI

                lmp = LammpsASELibrary(
                    working_directory=None,
                    cores=1,
                    comm=MPI.COMM_SELF,
                    logger=None,
                    log_file=None,
                    library=None,
                    disable_log_file=True,
                )
            else:
                lmp = LammpsASELibrary(
                    working_directory=None,
                    cores=1,
                    comm=None,
                    logger=None,
                    log_file=None,
                    library=None,
                    disable_log_file=True,
                )
        else:
            close_lmp_after_calculation = False

        # Run function
        result = funct(lmp=lmp, *args, **kwargs)

        # Close temporary LAMMPS instance
        if close_lmp_after_calculation:
            lmp.close()
        return result

    return funct_return

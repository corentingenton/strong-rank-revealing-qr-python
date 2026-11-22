import ctypes
import os
import sys


def list_mkl_routines():
    """List available routines in MKL library."""
    # Find MKL
    lib_path = os.path.join(sys.prefix, 'Library', 'bin')
    mkl_path = os.path.join(lib_path, 'mkl_rt.2.dll')

    if not os.path.exists(mkl_path):
        raise OSError(f"MKL not found at {mkl_path}")

    try:
        # Add DLL directory to PATH
        os.add_dll_directory(lib_path)
    except (AttributeError, OSError):
        pass

    # Load MKL
    mkl = ctypes.CDLL(mkl_path)

    # Get all attributes
    routines = []
    for attr in dir(mkl):
        # Look for LAPACK-style names (all caps or lowercase with underscore)
        if (attr.isupper() and len(attr) > 3) or '_' in attr:
            routines.append(attr)

    return sorted(routines)


# Print all routines containing "GEQP" (QR-related)
routines = list_mkl_routines()
print("Available QR-related routines:")
for routine in routines:
    if 'GEQP' in routine.upper():
        print(routine)
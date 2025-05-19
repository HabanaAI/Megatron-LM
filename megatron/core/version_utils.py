# Copyright (C) 2025 Intel Corporation

import subprocess
import warnings

from packaging import version

CURRENTLY_VALIDATED_GAUDI_VERSION = version.parse("1.21.0")


def check_gaudi_version():
    """
    Checks whether the versions of Gaudi and drivers have been validated for the current version of Megatron-LM.
    """
    # Check the version of habana_frameworks
    habana_frameworks_version_number = get_habana_frameworks_version()
    if (
        habana_frameworks_version_number.major != CURRENTLY_VALIDATED_GAUDI_VERSION.major
        or habana_frameworks_version_number.minor != CURRENTLY_VALIDATED_GAUDI_VERSION.minor
    ):
        warnings.warn(
            f"Megatron-LM v{CURRENTLY_VALIDATED_GAUDI_VERSION} has been validated for Gaudi v{CURRENTLY_VALIDATED_GAUDI_VERSION} but habana-frameworks v{habana_frameworks_version_number} was found, this could lead to undefined behavior!"
        )

    # Check driver version
    driver_version = get_driver_version()
    # This check is needed to make sure an error is not raised while building the documentation
    # Because the doc is built on an instance that does not have `hl-smi`
    if driver_version is not None:
        if (
            driver_version.major != CURRENTLY_VALIDATED_GAUDI_VERSION.major
            or driver_version.minor != CURRENTLY_VALIDATED_GAUDI_VERSION.minor
        ):
            warnings.warn(
                f"Megatron-LM v{CURRENTLY_VALIDATED_GAUDI_VERSION} has been validated for Gaudi v{CURRENTLY_VALIDATED_GAUDI_VERSION} but the driver version is v{driver_version}, this could lead to undefined behavior!"
            )
    else:
        warnings.warn(
            "Could not run `hl-smi`, please follow the installation guide: https://docs.habana.ai/en/latest/Installation_Guide/index.html."
        )


def get_habana_frameworks_version():
    """
    Returns the installed version of Gaudi.
    """
    output = subprocess.run(
        "pip list | grep habana-torch-plugin",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return version.parse(output.stdout.split("\n")[0].split()[-1])


def get_driver_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run(
        "hl-smi",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"ENABLE_CONSOLE": "true"},
    )
    if output.returncode == 0 and output.stdout:
        return version.parse(
            output.stdout.split("\n")[2].replace(" ", "").split(":")[1][:-1].split("-")[0]
        )
    return None


def is_habana_frameworks_min_version(min_version):
    """
    Checks if the installed version of `habana_frameworks` is larger than or equal to `min_version`.
    """
    if get_habana_frameworks_version() < version.parse(min_version):
        return False
    else:
        return True


def is_habana_frameworks_version(req_version):
    """
    Checks if the installed version of `habana_frameworks` is equal to `req_version`.
    """
    return (get_habana_frameworks_version().major == version.parse(req_version).major) and (
        get_habana_frameworks_version().minor == version.parse(req_version).minor
    )

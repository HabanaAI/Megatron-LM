# Copyright (C) 2025 Intel Corporation

import subprocess  # nosec
import warnings
from functools import lru_cache
from typing import Union

import torch
from packaging import version

CURRENTLY_VALIDATED_GAUDI_VERSION = version.parse("1.23.0")


def check_gaudi_version():
    """
    Checks whether the versions of Gaudi and drivers have been validated
    for the current version of Megatron-LM.
    """
    # Check the version of habana_frameworks
    habana_frameworks_version_number = get_habana_frameworks_version()
    if (
        habana_frameworks_version_number.major != CURRENTLY_VALIDATED_GAUDI_VERSION.major
        or habana_frameworks_version_number.minor != CURRENTLY_VALIDATED_GAUDI_VERSION.minor
    ):
        warnings.warn(
            f"Megatron-LM v{CURRENTLY_VALIDATED_GAUDI_VERSION} has been validated for "
            f"Gaudi v{CURRENTLY_VALIDATED_GAUDI_VERSION} "
            f"but habana-frameworks v{habana_frameworks_version_number} was found, "
            f"this could lead to undefined behavior!"
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
                f"Megatron-LM v{CURRENTLY_VALIDATED_GAUDI_VERSION} has been validated for "
                f"Gaudi v{CURRENTLY_VALIDATED_GAUDI_VERSION} "
                f"but the driver version is v{driver_version}, "
                f"this could lead to undefined behavior!"
            )
    else:
        warnings.warn(
            "Could not run `hl-smi`, please follow the installation guide: "
            "https://docs.habana.ai/en/latest/Installation_Guide/index.html."
        )


@lru_cache(maxsize=1)
@torch.compiler.assume_constant_result
def get_habana_frameworks_version():
    """
    Returns the installed version of habana-torch-plugin.
    """
    pip_process = subprocess.Popen(['pip', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    grep_process = subprocess.Popen(
        ['grep', 'habana-torch-plugin'],
        stdin=pip_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Get the output
    output, _ = grep_process.communicate()
    if not output:
        return version.parse("0.0.0.0")
    return version.parse(output.decode().split("\n")[0].split()[-1])


def get_intel_transformer_engine_version() -> Union[version.Version, str]:
    """
    Returns the installed version of intel_transformer_engine.
    """
    output = subprocess.run(
        "pip list | grep intel_transformer_engine",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not output.stdout:
        return version.parse("0.0.0.0")
    first_line_split = output.stdout.split("\n")[0].split()
    if len(first_line_split) > 1 and 'unknown' in first_line_split[1]:
        # This is the version when installed from source
        return "unknown"
    return version.parse(first_line_split[-1])


def get_driver_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run(
        "hl-smi",
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
    Checks if the installed version of `habana_frameworks` is larger than
    or equal to `min_version`.
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


def is_ite_min_version(min_version):
    """
    Checks if the installed version of `intel_transformer_engine`
    is larger than or equal to `min_version`.
    """
    ite_version = get_intel_transformer_engine_version()

    if isinstance(ite_version, str):
        warnings.warn(
            f"Got {ite_version} intel_transformer_engine version. "
            f"This typically means it was installed from source. "
            f"Unable to verify if it is >= {min_version} - assuming yes."
        )
        return True

    if ite_version < version.parse(min_version):
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


def is_ite_version(req_version):
    """
    Checks if the installed version of `intel_transformer_engine` is equal to `req_version`.
    """
    ite_version = get_intel_transformer_engine_version()

    if isinstance(ite_version, str):
        warnings.warn(
            f"Got {ite_version} intel_transformer_engine version. "
            f"This typically means it was installed from source. "
            f"Unable to verify if it equals {req_version} - assuming no."
        )
        return False

    return (ite_version.major == version.parse(req_version).major) and (
        ite_version.minor == version.parse(req_version).minor
    )

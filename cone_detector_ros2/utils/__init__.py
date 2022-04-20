"""
 Copyright (c) 2022 Ultralytics

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """
"""
utils/initialization
"""


def notebook_init(verbose=True):
    # Check system software and hardware
    print("Checking setup...")

    import os
    import shutil

    from utils.general import check_requirements, emojis, is_colab
    from utils.torch_utils import select_device  # imports

    check_requirements(("psutil", "IPython"))
    import psutil
    from IPython import display  # to display images and clear console output

    if is_colab():
        shutil.rmtree(
            "/content/sample_data", ignore_errors=True
        )  # remove colab /sample_data directory

    # System info
    if verbose:
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        display.clear_output()
        s = f"({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)"
    else:
        s = ""

    select_device(newline=False)
    print(emojis(f"Setup complete ✅ {s}"))
    return display

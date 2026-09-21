import os
# import glob
from pathlib import Path


def remove_files_from_curdir(path):
    pass
    if len(path) != 0:
        for i in glob.glob(os.path.join(path,"*")):
            # print(i)
            os.remove(i)

def check_and_mkdir(base, root="."):
    root = Path(root)

    folder = root / base

    if not folder.exists():
        folder.mkdir(parents=True)
        return folder

    if "_" in base:
        name, num = base.rsplit("_", 1)

        if num.isdigit():
            i = int(num)

            while True:
                new = root / f"{name}_{i}"

                if not new.exists():
                    new.mkdir(parents=True)
                    return new

                i += 1

    return folder
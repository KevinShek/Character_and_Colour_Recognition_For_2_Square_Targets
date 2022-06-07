import os
from pathlib import Path
import itertools

internal_folders = "../datasets/journal_1/distance_test/720p"

print(f"{internal_folders}".rsplit('/', 1)[-1])

# list_of_internal_folders = os.listdir(Path(internal_folders))

# for internal_folder in list_of_internal_folders:
#     data_dir = Path(f"{internal_folders}/{internal_folder}")
#     image_count = list(itertools.chain.from_iterable(data_dir.glob(pattern) for pattern in ('*.jpg', '*.png')))

#     print(image_count)
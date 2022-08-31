
import time
import os
from glob import glob, iglob
from pathlib import Path
import csv

directory = r"test_images"

def run_iglob():
    fu = list(iglob(os.path.join(directory, '**', '*.jpg', '*.png'), recursive=True))
    print(f"{fu}")
    print(f"{len(fu)}")
    image_count = sorted(list(itertools.chain.from_iterable(fu.glob(pattern) for pattern in ('*.jpg', '*.png'))))
    for name in image_count:
        # head, tail = ntpath.split(name)
        filename = Path(name)  # .stem removes the extension and .name grabs the filename with extension
        name_of_image = filename.stem
        # print(name_of_image)
        actual_character = f"{name_of_image}".rsplit('_', 5)[0]
        option = f"{name_of_image}".rsplit('_', 5)[-1]

        dict_of_colour_with_character = {'1': "red", '2': "green", '3': "blue", '4': "red", '5': "green", '6': "blue", '7': "red", '8': "green", '9': "blue", 
                        'A': "red", 'B': "green", 'C': "blue", 'D': "red", 'E': "green", 'F': "blue", 'G': "red", 'H': "green", 'I': "blue",'J': "red", 
                        'K': "green", 'L': "blue", 'M': "red", 'N': "green", 'O': "blue", 'P': "red", 'Q': "green", 'R': "blue", 'S': "red", 
                        'T': "green", 'U': "blue", 'V': "red", 'W': "green", 'X': "blue", 'Y': "red", 'Z': "green"}
        

        list_of_character = list(dict_of_colour_with_character.keys())
        list_of_colour = list(dict_of_colour_with_character.values())

        actual_colour = list_of_colour[list_of_character.index(str(actual_character))]

        results = [filename, actual_character, actual_colour]

        (directory / 'labels' if False else directory / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

        with open(directory / 'labels' / "{name_of_image}.txt", 'a') as csvfile:  # for testing purposes
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            filewriter.writerow(results)


if __name__ == '__main__':
    run_iglob()
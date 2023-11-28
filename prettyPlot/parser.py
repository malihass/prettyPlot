import os
import sys


def parse_input_file(input_filename="input"):
    if not os.path.isfile(input_filename):
        print(f"ERROR: No input file found with name : {input_filename}")
        sys.exit()

    # ~~~~ Parse input
    inpt = {}
    f = open(input_filename)
    data = f.readlines()
    for line in data:
        if ":" in line:
            key, value = line.split(":")
            inpt[key.strip()] = value.strip()
    f.close()

    return inpt

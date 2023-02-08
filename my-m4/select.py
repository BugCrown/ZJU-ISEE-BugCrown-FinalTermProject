import os
import pathlib
import sys
import numpy as np

def SelectArray(img_path, num):
    array = ""
    with open(img_path, "r") as f:
        for i in range(num + 1):
            array = f.readline()
        
    with open("temp.txt", "w+") as f:
        f.write(array)
        
if __name__ == "__main__":
    SelectArray(sys.argv[1], int(sys.argv[2]))
        
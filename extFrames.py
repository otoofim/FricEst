import matplotlib.pyplot as plt
import numpy as np
import re
import os
from IPython.display import Image 
import cv2
import numpy as np
import pandas as pd
import glob
from frictionDataloader import *
from UNet import *
import torchvision
from torchsummary import summary
from extractingFrames import *

dataset_arg = {
        "season": "Winter",
        "date": "23-03-2021",
        "hour": "15:58",
        "year": "2021"
}


loader = ExtractingFrames(dataset_arg)

loader.savingFrames()
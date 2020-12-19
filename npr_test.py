#!/usr/bin/env python

import npr
import numpy as np
x=np.random.normal(size=4096*128)  #信号长度  
npr.compare(x, 50, tap=12) #查看pfb的效果，见npr.py

# -*- coding: utf-8 -*-
from flask import current_app
import json
import time
import os
import gc
import hashlib
import signal
from io import BytesIO, StringIO
from itertools import combinations, permutations, product
import heapq
import tempfile
import csv
import numpy as np
import pandas as pd
import math
from collections import defaultdict
import re

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    matplotlib = None
    plt = None
    sns = None
try:
    import psutil
except Exception:
    psutil = None

from typing import Dict, List, Iterable, Union

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

print(f"[OPTIMIZER] PuLP disponible: {PULP_AVAILABLE}")

from threading import RLock, current_thread
from functools import wraps
import ctypes

_MODEL_LOCK = RLock()
active_jobs = {}
template_cfg = {}

# TODAS LAS FUNCIONES DEL LEGACY IMPLEMENTADAS CORRECTAMENTE
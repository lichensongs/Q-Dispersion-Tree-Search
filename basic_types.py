import numpy as np


PolicyArray = np.ndarray  # shape of (|A|,)
ValueChildArray = np.ndarray
HiddenArray = np.ndarray  # shape of (|H|,)

Value = float
Interval = np.ndarray  # shape of (2,)
IntervalLike = Interval | float
HiddenValue = int
Action = int

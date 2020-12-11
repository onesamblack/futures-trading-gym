import datetime
from typing import List, Tuple, Sequence, Union, Dict
import math
import pandas
import numpy

def round_to_nearest_increment(x :float, tick_size : float = 0.25) -> float:
  """
  rounds a price value to the nearest increment of `tick_size`
  
  Parameters
  -----------
  x : float
    the price value
  tick_size: float
    the size of the tick. E.G. for ES = 0.25, for CL = 0.01
  """
  val = round(tick_size * round(float(x)/tick_size), 2)
  if val < tick_size:
    return tick_size
  else:
    return val

def monotonicity(series: Union[pandas.Series, pandas.DataFrame, numpy.ndarray]) -> float:
  """
  Measures the monotonicity of a series on a -1 to 1 scale, with 1 being perfectly monotonic and
  -1 being anti-monotonic
  
  Parameters
  -----------
  series : Sequence
    a 
  
  """

  dirs = []
  for i, val in enumerate(series):
    if i == 0:   
      pass
    else:
      dif = val - series[i-1]
      if dif > 0:
        dirs.append(1)
      elif dif == 0:
        dirs.append(0)
      else:
        dirs.append(-1)
  return np.mean(dirs)


class TimeSeriesState:
  """
  A timeseries state is a representation of the current state. 
  The state may contain an arbitrary number of features, but it must
  contain at a minimum a timestamp, and a close price.

  The environment uses the `price` attribute to compute the current
  value of the security, as well as computing the profit per trade

  The environment uses the `ts` attribute to compute the duration of each
  trade as well as to enforce ordering, e.g. the timestamp of
  state(t-1) must be less than state(t)  

  ...

  Attributes
  ----------
  data : Sequence
    a sequence of records, such as a pandas.DataFrame, a np.ndarray, or a list of lists
    e.g. [[2020-01-01 01:01:59, 3290.25, 3290.50, .08, ... ]]
  close_price_identifier: Union[int, str]
    a int, string identifier for the close price in the data
  timestamp_identifier: Union[int, str]
    a int, string identifier for the timestamp in the data
  timestamp_format: str
    a format string using the directives specified in
    https://docs.python.org/3/library/time.html#time.strftime. Must be provided
    if the timestamp is not a datetime.datetime object

 

  """
  def __init__(self, data: Union[pandas.DataFrame, numpy.ndarray, List], close_price_identifier: Union[int,str] = None, 
               timestamp_identifier: Union[int, str] = None, timestamp_format: str = None):
    self.data = data
    if close_price_identifier:
      self.price = float(data[-1:][close_price_identifier])
    else:
      self.price = float(data[-1:]["close"])
    
    self.timestamp_format = "%Y-%m-%d %H:%M:%S" if not timestamp_format else timestamp_format

    if timestamp_identifier:
      self.ts = self._timestamp_to_py_time(data[-1:][timestamp_identifier])
    else:
      self.ts = self._timestamp_to_py_time(data[-1:]["time"])  
    self.current_position = None
    self.entry_time = None
    self.entry_price = None
  
  def _timestamp_to_py_time(self, ts: str):
    """converts the string ts to a datetime object 
    """
    val = list(ts)[0]
    if type(val) != datetime.datetime:
      return datetime.datetime.strptime(val, self.timestamp_format)
    else:
      return ts
  
  def set_current_position(self, pos: int, time: datetime.datetime, price: float):
    """
    Sets the current position of the environment in the state. This allows
    users to include the current trade in the training loop for the agent

    Parameters
    -----------
    pos: int
      one of [-1, 0, 1], -1 represents a short, 0 represents no position and 1 represents a long
    time: datetime.datetime
      the timestamp representing the acknowledged entry time of the trade
    price: float
      the entry price of the trade
    """
    self.current_position = pos
    self.entry_time = time
    self.entry_price = price
 
  def __str__(self):
    return f"timestamp: {self.ts}, price: {self.price}, current_position: {self.current_position}, entry_time: {self.entry_time}, entry_price: {self.entry_price}"
 

  def to_feature_vector(self, *args, **kwargs):
    """Users should override this. E.g. if your agent uses PyTorch tensors
    convert the current state and the data into a tensor object, etc.

    This depends on how your agent operates
    """
    pass
    
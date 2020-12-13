from gym_futures.envs.utils import *
import numpy as np
import pandas as pd
import datetime 
import torch


def test_round_to_nearest_increment():

  assert round_to_nearest_increment(100.63) == 100.75
  assert round_to_nearest_increment(100.62) == 100.50

def test_monotonicity():

  pos_range = np.arange(0, 10, 1)
  neg_range = np.arange(10,0, -1)
  zero_range = np.array([0,1,0,1,0,1,0])
  not_zero_range = np.array([0,1,0,1,0,1])

  assert monotonicity(pos_range) == 1
  assert monotonicity(neg_range) == -1
  assert monotonicity(zero_range) == 0 
  assert monotonicity(not_zero_range) != 0


def test_create_timeseriesstate_from_pandas_dataframe():
  df = pd.DataFrame([[datetime.datetime.now(), 1000, 1001, 10002], [datetime.datetime.now(), 1001, 1001, 10002]])
  state = TimeSeriesState(df, close_price_identifier=1, timestamp_identifier=0)

  assert type(state.ts) in [pd.Timestamp, datetime.datetime]
  assert state.price == 1001

def test_create_timeseriesstate_from_numpy_array():
  _ar = np.array([[datetime.datetime.now(), 1000, 1001, 10002], [datetime.datetime.now(), 1001, 1001, 10002]])
  print(_ar[-1:, 0])
  state = TimeSeriesState(_ar, close_price_identifier=1, timestamp_identifier=0)

  assert type(state.ts) in [pd.Timestamp, datetime.datetime]
  assert state.price == 1001


def test_create_timeseriesstate_from_list():
  _ar = [[datetime.datetime.now(), 1000, 1001, 10002], [datetime.datetime.now(), 1001, 1001, 10002]]
  state = TimeSeriesState(_ar, close_price_identifier=1, timestamp_identifier=0)

  assert type(state.ts) in [pd.Timestamp, datetime.datetime]
  assert state.price == 1001

def test_override_of_timeseries_state():
  class WindowedTimeSeries(TimeSeriesState):
    def __init__(self, data, window_size, **kwargs):
      super().__init__(data, **kwargs)
      self.window_size = window_size
      if len(data) != window_size:
        raise Exception("window size error")
   
    def to_feature_vector(self, *args, **kwargs):
      """
      A simple tensor
      """
      if type(self.data) == np.ndarray:
        return torch.tensor(np.array(self.data[:, 1:], dtype=numpy.float64))
      else:
        return torch.tensor(np.array(self.data.iloc[:,1:]))
    
  _ar = [[datetime.datetime.now(), 1000, 1001, 10002], [datetime.datetime.now(), 1001, 1001, 10002]]

  state = WindowedTimeSeries(_ar, window_size=2, close_price_identifier=1, timestamp_identifier=0)

  assert type(state.ts) in [pd.Timestamp, datetime.datetime]
  assert state.price == 1001

  assert type(state.to_feature_vector()) == torch.Tensor
  print(state.to_feature_vector())

  # test that an inccorect window size raises an exception 
  try:
    state_exception = WindowedTimeSeries(_ar, window_size=10)
    assert False
  except:
    assert True



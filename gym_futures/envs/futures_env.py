import gym
import json
from pathlib import Path
from gym import error, spaces, utils
from gym.utils import seeding
from uuid import uuid4


class FuturesEnv(gym.Env):
  """
  A gym for training futures trading RL environments.

  The futures market is different than a typical stock trading environment, in that
  contracts move in fixed increments, and each increment (tick) is worth a variable
  amount depending on the contract traded.

  This environment is designed for a single contract - for a single security type.
  Scaling the agent trained is only a matter of scaling up the order size (within reasonable)
  limits.

  Accompanying this environment is the concept of a TimeSeriesState, which is a variable
  2-d length window of all the features representing the state, or something that has
  additional dimensions (e.g a stack of scalograms). See `TimeSeriesState` for more
  details and examples

  This environment accepts 3 different actions at any time state:
    0 - a buy
    1 - a hold (no action)
    2 - a sell
  
  The environment does not allow for more than one contract to be traded at a time.
  If a buy action is submitted with an existing long trade, the action defaults to
  no action.

  You can add the current position to the next state sent to the agent by setting
  `add_current_trade_information_to_state` to True.  

  This environment can also simulate the probabilistic dynamics of actual market trading
  through the `fill_probability` and long/short probability vectors. Occasionally,
  an order will not fill, or will fill at a price that differs from the intended
  price. See `generate_random_fill_differential` for usage details. If deterministic
  behavior is desired, do not supply these arguments
  

  The standard reward function is the net_profit
  where net_profit is equal to
       ((entry_price - exit_price) / tick_size) * value_per_tick) 
      - execution_cost_per_order * 2 

  It's likely that you will want to use a more complex reward function.
  In this case, subclass this environment and overwrite `get_reward()`


  Attributes
  ----------
  states: Sequence[TimeSeriesState]
    a sequence of `TimeSeriesState` objects
  value_per_tick: float
    the value per 1 tick movement. E.g. 1 tick movement for ES is 12.50, for CL is 10
  tick_size: float
    the minimum movement in the price, or tick size. E.g. ES = 0.25, CL = 0.01
  fill_probability: float
    the probability of filling a submitted order. Defaults to 1
  long_values: List[float]
    a list of values that represent possible differentials from the
    intended fill price for buy orders
  long_probabilities: List[float]
    the probability distribution for the possible differentials
    specified in long_values. the length must equal the length of long_values and the sum 
    of the probabilities must equal 1
  short_values: List[float]
    a list of values that represent possible differentials from the
    intended fill price for sell_orders
  short_probabilities: List[float]
    the probability distribution for the possible differentials
    specified in short_values. the length must equal the length of
    short_values and the sum of the probabilities must equal 1
  execution_cost_per_trade: float
    the cost of executing 1 buy or sell order. Include the cost
    per 1 side of the trade, the calculation for net profit
    accounts for 2 orders
  add_current_position_to_state: bool
    adds the current position to the next state. Default: False
  log_dir: str
    a str or Path representing the directory to 
    render the results of the epoch. see `render` will generate
    output metrics to tensorflow
  """

  metadata = {'render.modes': ['human']}

  def __init__(self, states: Sequence[TimeSeriesState], value_per_tick: float, 
               tick_size: float, fill_probability: float = 1., long_values: List[float] = None, 
               long_probabilities: List[float] = None, short_values: List[float] = None, 
               short_probabilities: List[float] = None, execution_cost_per_order=0.,
               add_current_position_to_state: bool = False, 
               log_dir: str = f"logs/futures_env/{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')}"):
 
    self.states = states
    self.limit = len(self.states)
    self.value_per_tick = value_per_tick
    self.tick_size = tick_size
    self.long_values = long_values
    self.long_probabilities  = long_probabilities
    self.short_values = short_values
    self.short_probabilities = short_probabilities
    self.can_generate_random_fills = all([self.long_values, self.long_probabilities, self.short_values, self.short_probabilities])
    self.fill_probability = fill_probability
    self.execution_cost_per_order = execution_cost_per_order
    self.add_current_position_to_state = add_current_position_to_state
    self.log_dir = log_dir
    self.done = False
    self.current_index = 0

    self.current_price = None
    # attributes to maintain the current position
    self.current_position = 0
    self.last_position = 0
    
    self.entry_time = None
    self.entry_id = None
    self.entry_price = None

    self.exit_time = None
    self.exit_id = None
    self.exit_price = None

    # episode attributes
    self.total_reward = 0
    self.total_net_profit = 0
    self.orders = []
    self.trades = []
    self.episode = 0
    self.feature_data = []
    
    Path(self.log_dir).mkdir(parents=True, exist_ok=True)

  def buy(self, state: TimeSeriesState):
    """Creates a buy order"""
    if self.current_position == 1:
      # does not perform a buy order
      pass
    elif self.current_position == -1:
      self.last_position = self.current_position
      self.current_position = 0

      self.exit_price = self.generate_random_fill_differential(state.price, 1)
      self.exit_time = state.ts
      self.exit_id = str(uuid4())
      self.orders.append([self.exit_id, str(state.ts), self.exit_price, 1, state])

    elif self.current_position == 0:
      self.last_position = self.current_position
      self.current_position = 1
      self.entry_price = self.generate_random_fill_differential(state.price, 1)
      self.entry_time = state.ts
      self.entry_id = str(uuid4())
      self.orders.append([self.entry_id, str(state.ts), self.entry_price, 1, state])

  def sell(self, state: TimeSeriesState):
    """generates a sell order"""
    if self.current_position == -1:
      # does not perform a sell
      pass

    elif self.current_position == 1:
      self.last_position = self.current_position
      self.current_position = 0
      self.exit_price = self.generate_random_fill_differential(state.price, -1)
      self.exit_time = state.ts
      self.exit_id = str(uuid4())
      self.orders.append([self.exit_id, str(state.ts), self.exit_price, -1, state])

    elif self.current_position == 0:
      self.last_position = self.current_position
      self.current_position = -1
      self.entry_price = self.generate_random_fill_differential(state.price, -1)
      self.entry_time = state.ts
      self.entry_id = str(uuid4())
      self.orders.append([self.entry_id, str(state.ts), self.entry_price,-1, state])

  def get_reward(self, state: TimeSeriesState) -> float:
    """
    This environments default reward function. Override this class and method for a custom reward function
    """
    net_profit = 0
    if any([all([self.current_position == 0, self.last_position == 0]),
        all([self.current_position == 1, self.last_position == 0]),
        all([self.current_position == -1, self.last_position == 0])]):
      # no reward for no action taken or holding a position only receive rewards for closing a trade
      return net_profit

    else:
      if all([self.current_position == 0, self.last_position == 1]):
        # closed a long
        dif =  round((self.exit_price - self.entry_price),2)
      elif all([self.current_position == 0, self.last_position == -1]):
        # closed a short
        dif = - round((self.exit_price - self.entry_price),2)
      n_ticks = math.ceil(dif / self.tick_size)
      gross_profit = n_ticks * self.value_per_tick
      net_profit = gross_profit - (2*self.execution_cost_per_order)
    
    self.total_reward += net_profit
    return net_profit
        
   
  def step(self, action):
    """
    This mimics OpenAIs training cycle, where the agent produces an action, and the action is provided to the step function of the environment.
    The environment will return the expected (next_state, reward, done, info) tuple

        _s == s' (next state)
         s == s (the current state that the action is for)

    """
    _s, s = self._get_next_state()
    if self.done:
      return (None, None, self.done, None)

    current_state_price = s.price
    next_state_price = _s.price

    if action == 0:
      # a buy action signal is received
      if self.current_position == 1:
        # a buy is recommended whilst in a long - no action
        reward = self.get_reward(s) 
        info = {"message": "hold - a buy was recommended while in an open long position"}
        return (_s, reward, self.done, info)
      if self.current_position == 0:
        # a buy is recommended by the agent whilst no position - creating a long
        # this fills with pr(fill) == self.fill_probability
        if np.random.choice(a=[0,1], 
                            size=1, 
                            p=[1-self.fill_probability, self.fill_probability])[0] == 1:
          
          self.buy(s)

          reward = self.get_reward(s)

          info = {
              "message": f"timestamp: {str(self.entry_time)}, long trade attempted at: {current_state_price}, filled at: {self.entry_price}"
          }
          return (_s, reward, self.done, info)
        else:
          info= {
              "message": "a long was recommended, but was not filled given the current fill probability"
          }
          return(_s, reward, self.done, info)


      if self.current_position == -1:
        # a buy is recommended by the agent whilst in a sell. 
        # This closes a short

        self.buy(s)  

        reward = self.get_reward(s)

        net_profit = reward

        info = {
            "message": f"timestamp: {str(s.ts)}, short closed from {self.entry_price} to {self.exit_price} - total profit: {net_profit}"
        }

        self._close_position(reward, net_profit)

        return (_s, reward, self.done, info)

    elif action == 1:
      # no action recommended
      reward = self.get_reward(s)
      info = {"message": "no action performed"} 
      return (_s, reward, self.done, info)
    
    
    elif action == 2:
      # a sell signal is received
      if self.current_position == 1:
        # a sell is recommended by the agent whilst in a buy. 
        # This closes a long

        self.sell(s)  

        reward = self.get_reward(s)

        net_profit = reward

        info = {
            "message": f"timestamp: {str(s.ts)}, long closed from {self.entry_price} to {self.exit_price} - total profit: {net_profit}"
        }

        self._close_position(reward, net_profit)

        return (_s, reward, self.done, info)

      if self.current_position == 0:
        # a sell is recommended by the agent whilst no position - creating a short
        # this fills with pr(fill) == self.fill_probability
        if np.random.choice(a=[0,1], 
                            size=1, 
                            p=[1-self.fill_probability, self.fill_probability])[0] == 1:
          
          self.sell(s)

          reward = self.get_reward(s)

          info = {
              "message": f"timestamp: {str(self.entry_time)}, short trade attempted at: {current_state_price}, filled at: {self.entry_price}"
          }
          return (_s, reward, self.done, info)       
        
        else:
          info = {
              "message": "a long was recommended, but was not filled given the current fill probability"
          }
          return(_s, 0, self.done, info)

      if self.current_position == -1:
        # a sell is recommended whilst in a short - no action
        reward = self.get_reward(s) 
        info = info = {"message": "hold - a sell was recommended while in an open short position"}
        return (_s, reward, self.done, info)

  def reset(self, e):
    self.done = False
    self.current_index = 0

    self.current_price = None
    # attributes to maintain the current position
    self.current_position = 0
    self.last_position = 0
    
    self.entry_time = None
    self.entry_id = None
    self.entry_price = None

    self.exit_time = None
    self.exit_id = None
    self.exit_price = None

    # episode attributes
    self.total_reward = 0
    self.total_net_profit = 0
    self.orders = []
    self.trades = []
    self.feature_data = []
    
    self.episode = e
    return self.states[0]
  
  def render(self):
    """
    As the result of each episode, the render method will:
      1. generate distributions of the duration of each trade and the profit/loss for each trade
      2. generate pnl metrics (expectancy, win rate, etc)

    """
    self._generate_episode_graphs()
    metrics = self.generate_episode_metrics()

    return self.total_reward, metrics



  def _close_position(self, reward: float, net_profit:float):
    """resets the internal state for the position"""

    duration = (self.exit_time - self.entry_time).total_seconds()
    trade_id = str(uuid4())
    if all([self.current_position == 0, self.last_position == -1]):
      # closed a short
      self.trades.append([trade_id, "short", self.entry_id, self.exit_id, net_profit, reward, duration])
    elif all([self.current_position == 0, self.last_position == 1]):
      # closed a short
      self.trades.append([trade_id, "long", self.entry_id, self.exit_id, net_profit, reward, duration])

    self.current_position = 0
    self.last_position = 0
    
    self.entry_time = None
    self.entry_id = None
    self.entry_price = None

    self.exit_time = None
    self.exit_id = None
    self.exit_price = None

  def get_episode_data(self, return_states:bool = False) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Returns metadata of the trades that occurred in the episode

    Returns
    -------
    dataframes: Tuple[pandas.DataFrame]
      (orders, trades)
      orders: all of the orders that occurred in the episode
      trades: all of the trades that occurred in the episode
    """
    order_cols = ['order_id', 'timestamp', 'price', 'type']
    # does not include state in the order_df
    order_df = pd.DataFrame(np.array(self.orders)[:, :4])
    order_df.columns = order_cols

    trade_cols = ['trade_id', 'trade_type', 'entry_order_id', 'exit_order_id', 'profit', 'reward', 'duration']
    trade_df = pd.DataFrame(self.trades)
    trade_df.columns  = trade_cols

    return order_df, trade_df


  def generate_random_fill_differential(self, intent_price: float, trade_type:int) -> float:
    """
    Generates a random fill differential according to a 
    user specified probability distribution.

    In live market trading, the actual entry price intended will differ
    from the actual price the order fills. This 'differential' can be
    modeled with a probability distribution. 
    
    The actual entry_price will be equal to
    
    Price(entry) = differential + intent_price

    the actual differential is obtained by sampling 
    from a given probability distribution 

    for example, from observation, the fill differential (price_filled - intended_price)
    can take on the following values:
      [-0.5, -0.25, 0, 0.25, 0.50]
    with the following probabilities:
      [0.10, 0.25, 0.30, 0.30, 0.05]

    Parameters
    -----------
    intent_price: float
      the intended trade price 
    trade_type: int
      one of [-1, 1] representing a [sell, buy] respectively


    Returns
    -----------
    price: float
      the price + differential representing the "filled" order price

    """

    
    if not self.can_generate_random_fills:
      return intent_price
    else:
      if trade_type == 1:
        price = round_to_nearest_increment(intent_price 
                                           + np.random.choice(a=self.long_values, size=1, p=self.long_probabilities)[0], 
                                           self.minimum_price_movement) 
        
        return price
      if trade_type == -1:
        price = round_to_nearest_increment(intent_price 
                                           + np.random.choice(a=self.short_values, size=1, p=self.short_probabilities)[0], 
                                           self.minimum_price_movement)
    
        return price
   

  
  def generate_episode_metrics(self, as_file: bool=True) -> dict:
    """
    Generates typical session metrics such as win rate, loss rate, expectancy, etc.

    Parameters
    ----------
    as_file: bool
      saves metrics to a json file. Defaults to true


    Returns
    --------
    dict
      a dictionary containing all the metrics.
      The runnning pnl is returned as a list (for visualizations)
    """
    orders, trades = self.get_episode_data()

    long_df = trades[trades["trade_type"] == "long"]
    short_df = trades[trades["trade_type"] == "short"]
    all_trades = trades
    
    running_pnl = []
    pnl_val = 0
    for i, row in trades.iterrows():
       pnl_val += row['profit']
       running_pnl.append(pnl_val)
  
    total_longs = len(long_df)
    l_tnp =  sum(long_df['profit']) if total_longs > 0 else 0
    l_wins = len(long_df[long_df['profit'] >= 0])
    l_losses = len(long_df[long_df['profit'] < 0])
    l_win_pl = sum(long_df['profit'][long_df['profit'] >= 0]) if l_wins > 0 else 0
    l_loss_pl = sum(long_df['profit'][long_df['profit'] < 0]) if l_losses > 0 else 0
    l_max_win = np.max(long_df['profit'][long_df['profit'] >= 0]) if l_wins > 0 else 0
    l_max_loss = np.min(long_df['profit'][long_df['profit'] < 0]) if l_losses > 0 else 0 
    l_profit_factor = abs(l_win_pl) / abs(l_loss_pl) if l_loss_pl != 0 else 1.
    l_win_avg = np.mean(long_df['profit'][long_df['profit'] >= 0])
    l_loss_avg = np.mean(long_df['profit'][long_df['profit'] < 0])
    l_win_pct = 0 if total_longs == 0 else l_win_pl / total_longs

    total_shorts = len(short_df)
    s_wins = len(short_df[short_df['profit'] >= 0])
    s_losses = len(short_df[short_df['profit'] < 0])
    s_max_win = np.max(short_df['profit'][short_df['profit'] >= 0]) if s_wins > 0 else 0
    s_max_loss = np.min(short_df['profit'][short_df['profit'] <  0]) if s_losses > 0 else 0
    s_win_pl = sum(short_df['profit'][short_df['profit'] >= 0]) if s_wins > 0 else 0
    s_loss_pl = sum(short_df['profit'][short_df['profit'] <  0]) if s_losses > 0 else 0
    s_win_avg = np.mean(short_df['profit'][short_df['profit'] >= 0]) if s_wins > 0 else 0
    s_loss_avg = np.mean(short_df['profit'][short_df['profit'] <  0]) if s_losses > 0 else 0
    s_profit_factor = abs(s_win_pl) / abs(s_loss_pl) if s_loss_avg  !=0 else 1.
    s_tnp = sum(short_df['profit']) if total_shorts > 0 else 0
    s_win_pct = 0 if total_shorts == 0 else s_win_pl / total_shorts

    tnp = sum(all_trades['profit'])
    wins = all_trades[all_trades['profit'] >= 0]
    losses = all_trades[all_trades['profit'] < 0]
    total_trades = len(all_trades)
    max_win = np.max(all_trades['profit'][all_trades['profit'] >= 0])
    max_loss = np.min(all_trades['profit'][all_trades['profit'] < 0])
    payout_ratio = max_win / max_loss
    win_pl = sum(all_trades['profit'][all_trades['profit'] >= 0])
    win_avg =  np.mean(all_trades['profit'][all_trades['profit'] >= 0])
    loss_pl = sum(all_trades['profit'][all_trades['profit'] < 0])
    loss_avg = np.mean(all_trades['profit'][all_trades['profit'] < 0])
    profit_factor = abs(win_pl) / abs(loss_pl)
    win_percentage = (len(wins) / total_trades) * 100

    intraday_low = np.min(running_pnl)
    expectancy = (win_percentage * win_avg) - ((1-win_percentage) * loss_avg)

    
    vars_ = {
      'tnp': tnp, 
      'intraday_low': intraday_low,
      "profit_factor" :profit_factor,
      "win_percentage" :win_percentage, 
      "avg_loss" : loss_avg,
      'intraday_low' : intraday_low,
      'payout_ratio' :payout_ratio,
      'max_win' : max_win,
      'win_avg' :  win_avg,
      'max_loss' : max_loss,
      'expectancy' :  expectancy,
      'loss_avg' : loss_avg,
      's_tnp' : s_tnp,
      's_profit_factor' : s_profit_factor,
      'total_shorts' : total_shorts,
      's_win_percentage' : s_win_pct,
      's_max_win' : s_max_win,
      's_max_loss' : s_max_loss,
      's_winw_avg' : s_win_avg,
      's_loss_avg' :  s_loss_avg,
      'l_tnp' : l_tnp,
      'l_profit_factor'  :l_profit_factor,
      'total_longs' : total_longs,
      'l_win_percentage' : l_win_pct,
      'l_max_win' : l_max_win,
      'l_max_loss' : l_max_loss,
      'l_win_avg' : l_win_avg,
      'l_loss_avg' : l_loss_avg,
      'pnl_monotonicity': monotonicity(running_pnl),
      'running_pl': running_pnl
    }
    Path(f"{self.log_dir}/metrics").mkdir(parents=True, exist_ok=True)
    if as_file:
      with open(f"{self.log_dir}/metrics/metrics_episode_{str(self.episode)}.json", "w+") as _file:
        _file.write(json.dumps(vars_, indent=4))
    return vars_


  def _generate_episode_graphs(self):
    import matplotlib.pyplot as plt
    orders, trades =  self.get_episode_data()
    
    short_df = trades[trades["trade_type"] == "short"]
    long_df = trades[trades["trade_type"] == "long"]
    durations = trades["duration"]
    
    #trade durations
    fig1 = plt.hist(durations, bins=20)
    Path(f"{self.log_dir}/img").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{self.log_dir}/img/duration_distribution_episode_{self.episode}.png")
    plt.clf()
    plt.hist(long_df["profit"], bins=20, alpha=0.5, label="long")
    plt.hist(short_df["profit"], bins=20, alpha=0.5, label="short")
    plt.savefig(f"{self.log_dir}/img/profit_loss_distribution_episode_{self.episode}.png")
    plt.clf()


  def _get_next_state(self):
    current_state = self.states[self.current_index]
    self.current_index += 1
    if self.current_index  <= self.limit -1 :
      next_state = self.states[self.current_index]
      self.current_price = current_state.price
      if current_state.ts > next_state.ts:
        raise Exception("the time stamp of the current state is greater than the next state")
      # adds current position to the next state
      if self.add_current_position_to_state:
        next_state.set_current_position(self.current_position, self.entry_time, self.entry_time)
      return (next_state, current_state)
    else:
      self.done = True
      return (None, current_state) 
import gym
import pathlib
from gym import error, spaces, utils
from gym.utils import seeding


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
  render_to_file: Union[str, pathlib.Path]
    if specified, a str or Path representing the directory to 
    render the results of the epoch. see `render` for details on what 
    is generated for each epoch. Default: None.
  """

  metadata = {'render.modes': ['human']}

  def __init__(self, states: Sequence[TimeSeriesState], value_per_tick: float, 
               tick_size: float, fill_probability: float = 1., long_values: List[float] = None, 
               long_probabilities: List[float] = None, short_values: List[float] = None, 
               short_probabilities: List[float] = None, execution_cost_per_order=0.,
               add_current_trade_information_to_state: bool = False, render_to_file: Union[str, pathlib.Path] = None):
 
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
    self.add_current_trade_information_to_state = add_current_trade_information_to_state
    self.render_to_file = render_to_file
    self.done = False
    self.current_index = 0
    self.current_price = None
    self.position = 0
    self.ts_last_trade = None
    self.price_last_trade = None
    self.profit_last_trade = 0.
    self.total_reward = 0
    self.total_net_profit = 0
    self.buys = []
    self.sells = []
    self.longs = []
    self.shorts = []
    self.trade_durations = []
    self.feature_data = []
   
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
      if self.position == 1:
        # a buy action is recommended by the agent whilst currently in a long -> a hold
        info = {"message": "hold - a buy was recommended while in an open long position"}
        reward = self.get_reward() 
        return (_s, reward, self.done, info)
      if self.position == 0:
        # a buy is recommended by the agent whilst no position - creating a long
        # this fills with pr(fill) == self.fill_probability
        if np.random.choice(a=[0,1], 
                            size=1, 
                            p=[1-self.fill_probability, self.fill_probability])[0] == 1:
          self.time_last_trade = s.ts
          self.position = 1
          trade_price = self.generate_random_fill_differential(current_state_price, 1)
          self.price_last_trade = trade_price
          reward = self.get_reward()
          info = {
              "message": "long trade entered at attempt:{}, fill:{}, {}".format(current_state_price, trade_price, self.time_last_trade)
          }
          self.buys.append([str(s.ts), trade_price])
          return (_s, reward, self.done, info)
        else:
          info= {
              "message": "a long was recommended, but was not filled given the current fill probability"
          }
          return(_s, 0, self.done, info)


      if self.position == -1:
        # a buy is recommended by the agent whilst in a sell. This closes a short
          
        _time_last_trade = self.time_last_trade
        time_current_trade = s.ts

        total_elapsed_seconds = (time_current_trade - _time_last_trade).total_seconds()

        # dif will be + if the current_state_price is BELOW the previously recorded
        # price. This will result in a positive profit movement corresponding with the
        # trade direction
        plt = self.price_last_trade
        current_state_price = self.generate_random_fill_differential(current_state_price, 1)
        dif = round((plt - curent_state_price),2)
        n_ticks = math.ceil(dif / self.minimum_price_movement)
        gross_profit = n_ticks * self.contract_multiplier
        net_profit = gross_profit - (2*self.commission_per_trade)
        
        if net_profit < 0:
          # if the net_profit is negative, apply the full negative reward to `dissuade`
          reward = net_profit 
        else:
          # apply the time discounted net profit
          # this is to encourage short term high profit trades vs long term holding
          try:
            reward = net_profit * ((n_ticks*5)*((1/2)**total_elapsed_seconds))
          except OverflowError:
            reward = net_profit
          

        self._close_position(reward, net_profit)

        info = {
            "message": "short closed from  {} to {} - total profit: {}, {}".format(plt, 
                                                                                   current_state_price,
                                                                                   net_profit, 
                                                                                   s.ts)
        }
        self.buys.append([str(s.ts), s.price])
        self.shorts.append([str(s.ts), str(_time_last_trade), plt, current_state_price, net_profit, reward, total_elapsed_seconds])
        self.trade_durations.append(total_elapsed_seconds)
        return (_s, reward, self.done, info)

    elif action == 1:
      reward = self.reward()
      info = {"message": "no action performed"} 
      return (_s, reward, self.done, info)
    
    
    elif action == 2:
      # a sell signal is received
      if self.position == 1:
      # a sell is recommended whilst in a long - this is a long close
        _time_last_trade = self.time_last_trade
        time_current_trade = s.ts

        total_elapsed_seconds = int((time_current_trade - _time_last_trade).total_seconds())

        # dif will be + if the current_state_price is ABOVE the previously recorded
        # price. This will result in a positive profit movement corresponding with the
        # trade direction
        plt = float(self.price_last_trade)
        current_state_price = self.generate_random_fill_differential(float(current_state_price), -1)
        dif = round((float(current_state_price) - plt), 2)
        n_ticks = math.ceil(dif / self.minimum_price_movement)
        gross_profit = n_ticks * self.contract_multiplier

        net_profit = gross_profit - (2*self.commission_per_trade)
        if net_profit < 0:
          # if the net_profit is negative, apply the full negative reward to `dissuade`
          reward = net_profit
        else:
          # apply the time discounted net profit
          # this is to encourage short term high profit trades vs long term holding
          try:
            reward = net_profit * ((n_ticks*5)*((1/2)**total_elapsed_seconds))
          except OverflowError:
            reward = net_profit
        

        self.position = 0
        self.price_last_trade = None
        self.time_last_trade = None
        info = {"message": "long closed from {} to {} - total profit: {}, {}".format(plt, current_state_price, net_profit, s.ts)}
        self.total_reward += reward
        self.total_net_profit += net_profit
        self.sells.append([s.ts_string, s.price])
        self.longs.append([s.ts_string, str(_time_last_trade), plt, current_state_price, net_profit, reward, total_elapsed_seconds])
        self.trade_durations.append(total_elapsed_seconds)
        return (_s, reward, self.done, info)

      if self.position == 0:
        if np.random.choice(a=[0,1],size=1, p=[1-self.fill_probability, self.fill_probability])[0] == 1:
          # a sell is recommended by the agent whilst no position - creating a short
          self.time_last_trade = s.ts
          trade_price = self.generate_random_fill_differential(float(current_state_price), -1)
          self.price_last_trade = trade_price
          self.position = -1
          reward = 0
          info = "short trade entered at attempt:{}, fill:{}, {}".format(current_state_price, trade_price, self.time_last_trade)
          self.sells.append([s.ts_string, trade_price])
          return (_s, reward, self.done, info)
        else:
          info= "a short was recommended, but was not filled given the current fill probability"
          return(_s,0,self.done, info)

      if self.position == -1:
        
        reward = 0
          
        info = "no trade performed, no reward to calculate"
        return (_s, reward, self.done, info)

  def reset(self):
    self.done = False
    self.current_index = 0
    self.position = 0
    self.time_last_trade = None
    self.price_last_trade = None
    self.total_reward = 0
    self.total_net_profit = 0
    self.buys = []
    self.sells = []
    self.longs = []
    self.shorts = []
    self.trade_durations = []
    self.current_price = self.states[self.current_index].price
    # return the first state of the episode
    return self.states[0]
  
  def render(self):
    pass
  
  def close(self):
    pass

  def get_reward(self):
    pass


  def _close_position(self, reward, net_profit):
    self.position = 0
    self.ts_last_trade = None
    self.price_last_trade = None
    self.total_reward += reward
    self.total_net_profit += net_profit

  def get_episode_dataframes(self):
    """
    Returns dataframes of the trades that occurred in the episode

    Returns
    -------
    dataframes: Tuple[pandas.DataFrame]
      (buy_orders, sell_orders, longs, shorts, all_trades)
      buy_orders: all of the buys that occurred in the episode
      sell_orders: all of the sells that occurred in the episode
      longs: all 'long' trades.  a long is a trade that begins with a buy, ends with a sell
        Intended to profit from increases in price
      shorts: all 'short' trades, a short is a trade that enters with a sell, ends with a buy. 
        Intended to profit from decreases in price
    """
    bs_cols = ['time',  'close']
    ls_cols = ['time', 'entry_time', 'entry_price', 'exit_price', 'profit', 'reward', 'duration']
    buy_df = pd.DataFrame(self.buys)
    buy_df.columns = bs_cols
    sell_df = pd.DataFrame(self.sells)
    sell_df.columns = bs_cols

    long_df = pd.DataFrame(self.longs)
    long_df.columns = ls_cols
    long_df['trade_type'] = 'LONG'

    short_df = pd.DataFrame(self.shorts)
    short_df.columns = ls_cols
    short_df['trade_type'] = 'SHORT'

    all_trades = pd.concat([long_df, short_df])

    all_trades.sort_values(by='time', inplace=True)
    return buy_df, sell_df, long_df, short_df, all_trades


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
   

  
  def generate_episode_metrics(self, as_file=None):
    """
    Generates typical session metrics such as win rate, loss rate, expectancy, etc.
    """
    buy_df, sell_df, long_df, short_df, all_trades = self.get_episode_dataframes()
    
    running_pl = []
    pl_val = 0
    for i, row in  all_trades.iterrows():
       pl_val += row['profit']
       running_pl.append(pl_val)
  
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

    intraday_low = np.min(running_pl)
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
      'pnl_monotonicity': monotonicity(running_pl),
      'running_pl': running_pl
    }
    if as_file:
      with open(as_file, "w+") as _file:
        _file.write(json.dumps(vars_))
    return vars_


  def generate_episode_graphs(self,show=False):
    import matplotlib.pyplot as plt
    buy_signal, sell_signal, long_df, short_df, _=  self.get_episode_dataframes()

    #trade durations
    fig1 = plt.hist(self.trade_durations, bins=20)
    plt.savefig("img/trade_durations_{}.png".format(self.episode))
    plt.clf()
    plt.hist(long_df['profit'], bins=20, alpha=0.5, label='long')
    plt.hist(short_df['profit'], bins=20, alpha=0.5, label='short')
    plt.savefig("img/long_short_profit_{}.png".format(self.episode))
    plt.clf()


  def _get_next_state(self):
    current_state = self.state[self.current_index]
    self.current_index += 1
    if self.current_index  <= self.limit -1 :
      next_state = self.states[self.current_index]
      self.current_price = current_state.price
      if current_state.ts > next_state.ts:
        raise Exception("the time stamp of the current state is greater than the next state")
      # adds current position to the next state
      if self.add_current_trade_information_to_state:
        next_state.set_current_position(self.position, self.time_last_trade, self.price_last_trade)
      return (next_state, current_state)
    else:
      self.done = True
      return (None, current_state) 
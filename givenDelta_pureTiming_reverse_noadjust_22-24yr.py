import pandas as pd 
import numpy as np
from zads import tick, Instrument, trading_date_distance, trading_date_offset, trading_dates, option_symbols
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import datetime
from joblib import Parallel, delayed
import warnings
# from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings('ignore')
import time
from scipy.optimize import brentq
import os




def get_rollingDates(target_folder):
    dateslist = []
    for f in os.scandir(target_folder):
        if f.is_dir():
            y = f.name.split('-')[0]
            if (y < '2022')|(y > '2024'):
                # print(f.name)
                continue
            else:
                dateslist.append(f.name)
    sort_dateslist = sorted(dateslist)
    return sort_dateslist


def get_ETF_Data(date:str, inst_name:str = '510300.XSHG'):
    inst_obj = Instrument(inst_name, date, date)
    ETF_oneday = inst_obj.kline_day
    return ETF_oneday 


def get_parallel_ETF_Data(trading_date_list:list):
    list_ETF = Parallel(n_jobs = min(16,len(trading_date_list)), backend = 'loky', verbose = 0)(
        delayed(get_ETF_Data)(
            date
        ) for date in trading_date_list
    )
    ETF_daily = pd.concat(list_ETF)

    keep_cols = ['time','open','close']
    ETF_daily.drop(columns = set(ETF_daily.columns)-set(keep_cols),inplace = True)
    ETF_daily.set_index('time', inplace = True) 
    ETF_daily = ETF_daily.rename(columns = {'time':'date','close':'ETF_close','open':'ETF_open'})
    return ETF_daily
    


# 返回标的 min df
def get_strikes(ETF_daily:pd.DataFrame, start_date:str = '2022-02-11', end_date:str = '2024-01-10'): 
    ''' 300ETF 期权价格变动单位
    price_gaps = {
        3:0.05,
        5:0.1,
        10:0.25,
        20:0.5,
        50:1,
        100:2.5,
        200:5,
    }
    '''

    # 获取期权合约的虚一档行权价格
    for i in ETF_daily.index:
        if (ETF_daily.loc[i,'ETF_open']<=5)&(ETF_daily.loc[i,'ETF_open']>3):
            gap = 0.1
            # print('res',ETF_daily.loc[i,'ETF_open'], ETF_daily.loc[i,'ETF_open']%1)  
        elif (ETF_daily.loc[i,'ETF_open']<=10)&(ETF_daily.loc[i,'ETF_open']>5):
            gap = 0.25

        if (ETF_daily.loc[i,'ETF_open']*100%10) < 5: # 平值
            ETF_daily.loc[i,'atm_call'] = math.floor(ETF_daily.loc[i,'ETF_open']*10)/10   
        elif (ETF_daily.loc[i,'ETF_open']*100%10) >= 5:
            ETF_daily.loc[i,'atm_call'] = math.ceil(ETF_daily.loc[i,'ETF_open']*10)/10  
          
        ETF_daily.loc[i,'otm1_call'] = round(ETF_daily.loc[i,'atm_call']+gap,2)
        ETF_daily.loc[i,'otm2_call'] = round(ETF_daily.loc[i,'atm_call']+2*gap,2)
        ETF_daily.loc[i,'otm3_call'] = round(ETF_daily.loc[i,'atm_call']+3*gap,2)
        ETF_daily.loc[i,'otm4_call'] = round(ETF_daily.loc[i,'atm_call']+4*gap,2)
        ETF_daily.loc[i,'itm1_call'] = round(ETF_daily.loc[i,'atm_call']-gap,2)
        ETF_daily.loc[i,'itm2_call'] = round(ETF_daily.loc[i,'atm_call']-2*gap,2)
        ETF_daily.loc[i,'itm3_call'] = round(ETF_daily.loc[i,'atm_call']-3*gap,2)
        ETF_daily.loc[i,'itm4_call'] = round(ETF_daily.loc[i,'atm_call']-4*gap,2)
        ETF_daily.loc[i,'itm5_call'] = round(ETF_daily.loc[i,'atm_call']-5*gap,2)
        ETF_daily.loc[i,'itm6_call'] = round(ETF_daily.loc[i,'atm_call']-6*gap,2)

    ETF_daily = ETF_daily[(ETF_daily.index<=end_date)&(ETF_daily.index>=start_date)]

    ETF_daily.to_csv('noadjust_22-24_ETF_daily.csv')
    return ETF_daily




def get_option_daily(date, ETF_daily, signal_ttm:int = 57, open_minTime:int = 4, close_minTime:int = 229):
    path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_510300.XSHG_call_info_df.csv'
    option_traded = pd.read_csv(path, index_col=0)
    
    option_dict_list = [] 
    for column in ['atm_call','otm1_call','otm2_call','otm3_call','otm4_call','itm1_call','itm2_call','itm3_call','itm4_call','itm5_call', 'itm6_call']: 
        df = option_traded[(pd.to_numeric(option_traded['symbol'].str[-4:]) == 1000*ETF_daily.loc[date, column])]
        df = df.sort_values('time_to_maturity').copy()
        df.reset_index(drop=True, inplace=True)

        for i in df.index:
            if df.loc[i,'time_to_maturity'] >= signal_ttm:
                option_code = df.loc[i, 'order_book_id']
                symbol = df.loc[i, 'symbol']
                maturity_date = df.loc[i, 'maturity_date']
                time_to_maturity = df.loc[i, 'time_to_maturity']
                contract_multiplier = df.loc[i, 'contract_multiplier']
                option_type =  df.loc[i, 'option_type']
                break
    
        sub_path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_{option_code}_{symbol}_greeksSeries.csv'

        greeks_df = pd.read_csv(sub_path, index_col=0)
        open = greeks_df.loc[open_minTime,'option_price']
        close = greeks_df.loc[close_minTime,'option_price']
        close_delta = greeks_df.loc[close_minTime,'delta']
        target_greeks_dict = greeks_df.iloc[open_minTime].to_dict() # 一致
        target_greeks_dict['strike'] = round(int(target_greeks_dict['symbol'][-4:])*0.001,1)
        target_greeks_dict['open'] = open
        target_greeks_dict['close'] = close
        target_greeks_dict['close_delta'] = close_delta
        target_greeks_dict['maturity_date'] = maturity_date
        target_greeks_dict['ttm'] = time_to_maturity
        target_greeks_dict['contract_multiplier'] = contract_multiplier

        option_dict_list.append(target_greeks_dict)
   
    option_daily_df = pd.DataFrame(option_dict_list)
    option_daily_df['type'] = [option_type]*option_daily_df.shape[0]

    return option_daily_df



# # 生成一分钟3个期权信息的skew df
def get_parallel_option_daily(ETF_daily, trading_date_list:list, signal_ttm:int = 45, open_minTime:int = 4, close_minTime:int = 229):
    list_callDay = Parallel(n_jobs = min(16,len(trading_date_list)), backend = 'loky', verbose = 0)(
        delayed(get_option_daily)(
            date,
            ETF_daily,
            signal_ttm = signal_ttm,
            open_minTime = open_minTime,
            close_minTime = close_minTime
        ) for date in ETF_daily.index
    ) 

    option_daily_df = pd.concat(list_callDay)
    option_daily_df.rename({'trading_date':'time','contract_id':'inst', 'IV':'open_iv', 'delta':'open_delta','gamma':'open_gamma','vega':'open_vega','theta':'open_theta'}, axis = 1, inplace = True)# 别忘记  axis = 1


    keep_columns = ['time','inst','type','strike','symbol','open','close','maturity_date','contract_multiplier','time_to_maturity','open_iv','open_delta','close_delta','open_gamma','open_vega','open_theta']
    option_daily_df.drop(columns=set(option_daily_df.columns)-set(keep_columns), inplace=True)
    option_daily_df.reset_index(drop=True, inplace=True)
    option_daily_df = pd.merge(ETF_daily, option_daily_df,on = 'time')
    
    print('finish strike calculation!!', option_daily_df)
    # option_daily_df.to_csv('22-24_45_option_daily.csv')
    return option_daily_df





def get_option_current_month(date, ETF_daily, min_ttm:int = 5, open_minTime:int = 4, close_minTime:int = 229):
    path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_510300.XSHG_call_info_df.csv'
    option_traded = pd.read_csv(path, index_col=0)
    
    option_dict_list = [] 
    for column in ['atm_call','otm1_call','otm2_call','otm3_call','otm4_call','itm1_call','itm2_call','itm3_call','itm4_call','itm5_call','itm6_call']: 
        df = option_traded[(pd.to_numeric(option_traded['symbol'].str[-4:]) == 1000*ETF_daily.loc[date, column])]
        df = df.sort_values('time_to_maturity').copy()
        df.reset_index(drop=True, inplace=True)

        for i in df.index:
            if df.loc[i,'time_to_maturity'] >= min_ttm:
                option_code = df.loc[i, 'order_book_id']
                symbol = df.loc[i, 'symbol']
                maturity_date = df.loc[i, 'maturity_date']
                time_to_maturity = df.loc[i, 'time_to_maturity']
                contract_multiplier = df.loc[i, 'contract_multiplier']
                option_type =  df.loc[i, 'option_type']
                break
    
        sub_path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_{option_code}_{symbol}_greeksSeries.csv'

        greeks_df = pd.read_csv(sub_path, index_col=0)
        open = greeks_df.loc[open_minTime,'option_price']
        close = greeks_df.loc[close_minTime,'option_price']
        close_delta = greeks_df.loc[close_minTime,'delta']
        target_greeks_dict = greeks_df.iloc[open_minTime].to_dict() # 一致
        target_greeks_dict['strike'] = round(int(target_greeks_dict['symbol'][-4:])*0.001,1)
        target_greeks_dict['open'] = open
        target_greeks_dict['close'] = close
        target_greeks_dict['close_delta'] = close_delta
        target_greeks_dict['maturity_date'] = maturity_date
        target_greeks_dict['ttm'] = time_to_maturity
        target_greeks_dict['contract_multiplier'] = contract_multiplier

        option_dict_list.append(target_greeks_dict)
   
    option_daily_df = pd.DataFrame(option_dict_list)
    option_daily_df['type'] = [option_type]*option_daily_df.shape[0]

    return option_daily_df



# # 生成一分钟3个期权信息的skew df
def get_parallel_option_current_month(ETF_daily, trading_date_list, min_ttm:int = 5, open_minTime:int = 4, close_minTime:int = 229):
    list_callDay = Parallel(n_jobs = min(16,len(trading_date_list)), backend = 'loky', verbose = 0)(
        delayed(get_option_current_month)(
            date,
            ETF_daily,
            min_ttm = min_ttm,
            open_minTime = open_minTime,
            close_minTime = close_minTime
        ) for date in ETF_daily.index
    ) 

    option_daily_df = pd.concat(list_callDay)
    option_daily_df.rename({'trading_date':'time','contract_id':'inst', 'IV':'open_iv', 'delta':'open_delta','gamma':'open_gamma','vega':'open_vega','theta':'open_theta'}, axis = 1, inplace = True)# 别忘记  axis = 1

    keep_columns = ['time','inst','type','strike','symbol','open','close','maturity_date','contract_multiplier','time_to_maturity','open_iv','open_delta','close_delta','open_gamma','open_vega','open_theta']
    option_daily_df.drop(columns=set(option_daily_df.columns)-set(keep_columns), inplace=True)
    option_daily_df.reset_index(drop=True, inplace=True)
    # print(option_daily_df,'\n',option_daily_df.columns)
    option_daily_df = pd.merge(ETF_daily, option_daily_df,on = 'time')
    
    # print('finish strike calculation!!', option_daily_df)
    option_daily_df.to_csv('otm1_atm_option_current_month.csv')
    return option_daily_df




import matplotlib.pyplot as plt


def get_timing_signal(option_iv, otm_level = 'otm1_call', timing_path = '/home/zhangyan/option_skew/timing_signal.csv', start_date='2022-01-28', end_date = '2024-01-10'):
    timing_df = pd.read_csv(timing_path, index_col = 0)
    timing_df['date'] = pd.to_datetime(timing_df['date'])
    factor_df = timing_df[(timing_df['date']>=start_date)&(timing_df['date']<=end_date)]
    factor_df.reset_index(drop = True, inplace = True)
    
    
    for i in range(1, factor_df.shape[0]):
        date = str(factor_df.loc[i, 'date'])[:10]
        last_date = str(factor_df.loc[i-1, 'date'])[:10]
        
        option_oneday =  option_iv.loc[option_iv['time'] == date]
        last_option_oneday =  option_iv.loc[option_iv['time'] == last_date]
        # print()
        try:
            factor_df.loc[i,'ETF_open'] = option_oneday['ETF_open'].unique()
        except:
            print('error!!', date, option_oneday['ETF_open'])
            break
        factor_df.loc[i,'ETF_close'] = option_oneday['ETF_close'].unique()
        

        new_atm_call = option_oneday['atm_call'].values[0]
        maturity_date = option_oneday.loc[option_oneday['strike'] == float(new_atm_call),'maturity_date'].values[0]
        last_maturity_date = last_option_oneday.loc[last_option_oneday['strike'] == float(new_atm_call),'maturity_date'].values[0]

        if i == 1:
            factor_df.loc[i-1,'ETF_open'] = last_option_oneday['ETF_open'].unique()
            factor_df.loc[i-1,'ETF_close'] = last_option_oneday['ETF_close'].unique()
            factor_df.loc[i-1,'rollover'] = 0 
            factor_df.loc[i-1,'final_signal'] = 0
            factor_df.loc[i-1,'direction'] = factor_df.loc[i-1, 'isHedge_signal'] # -1

        factor_df.loc[i, 'rollover'] = 1 if (maturity_date !=  last_maturity_date) else 0   # 默认不发生移仓换月， 如果发生换合约 rollover = 1 

        if (factor_df.loc[i, 'isHedge_signal'] != 0)&(factor_df.loc[i-1, 'isHedge_signal'] == 0):  # 开仓
            factor_df.loc[i, 'final_signal'] = factor_df.loc[i, 'isHedge_signal']   # -1
            factor_df.loc[i, 'direction'] = factor_df.loc[i, 'isHedge_signal']    # -1
        elif (factor_df.loc[i, 'isHedge_signal'] == 0)&(factor_df.loc[i-1, 'isHedge_signal']!= 0):  # 平仓
            factor_df.loc[i, 'final_signal'] = -1*factor_df.loc[i-1, 'isHedge_signal']   # no -1
            factor_df.loc[i, 'direction'] = 0

        else: # 持仓或者空仓
            factor_df.loc[i, 'final_signal'] = 0 
            factor_df.loc[i, 'direction'] = factor_df.loc[i-1, 'direction'] 

        if i == 1:
            factor_df.loc[i,'final_signal'] = factor_df.loc[i, 'isHedge_signal'] # -1


    keep_col = ['date','ETF_open','ETF_close','isHedge_signal','final_signal','direction','rollover']
    factor_df.drop(columns = set(factor_df.columns)- set(keep_col), inplace = True)
    factor_df.rename(columns={'direction': 'final_direction'}, inplace=True)
    factor_df.to_csv(f'noadjust_22-24_{otm_level[:4]}_atm_timing_test.csv')
    return factor_df





# 开仓信息统计 
def get_open_infos(far_atm_delta, far_max_otm_delta, direction, ETF_open, ETF_pre_close, new_atm_open, new_max_otm_open, contract_multiplier, new_atm_delta, new_max_otm_delta, init_cash = 1000000, risk_expo = 0.2):
    otm_num = int(init_cash*(1+risk_expo)/(far_max_otm_delta*ETF_open*contract_multiplier))
    atm_num = int(init_cash/(far_atm_delta*ETF_open*contract_multiplier))
    far_cash_delta = direction*(far_max_otm_delta*otm_num - far_atm_delta*atm_num)*ETF_open*contract_multiplier
 
    while abs(far_cash_delta) > 200000:
        if ((direction == 1)&(far_cash_delta<0))|((direction == -1)&(far_cash_delta>0)):
            otm_num += 1
        else:
            otm_num -= 1
        far_cash_delta = direction*(far_max_otm_delta*otm_num - far_atm_delta*atm_num)*ETF_open*contract_multiplier
        
        # otm_num -= 1
        # far_cash_delta = direction*(far_max_otm_delta*otm_num - far_atm_delta*atm_num)*ETF_open*contract_multiplier

    # margin  保证金
    if direction == -1:
        max_otm_margin = otm_num*ETF_pre_close*contract_multiplier*0.12             # short max otm
        atm_margin = atm_num*new_atm_open*contract_multiplier                                   # long atm
    elif direction == 1:
        max_otm_margin = otm_num*new_max_otm_open*contract_multiplier               # long max otm
        atm_margin = atm_num*ETF_pre_close*contract_multiplier*0.12                             # short atm    
    total_margin = max_otm_margin + atm_margin # 无论买卖期权都交保证金, 不区分符号
    
    # greeks
    new_cash_delta = direction*(new_max_otm_delta*otm_num - new_atm_delta*atm_num)*ETF_open*contract_multiplier
    
    
    return otm_num, atm_num, total_margin, new_cash_delta, far_cash_delta



# 以实时价格 计算平仓margin
def get_margin_back(date, ETF_pre_close, last_str, option_current_date, last_option_current_date, contract_multiplier, last_contract_multiplier, option_commission:int = 3, open_minTime:int = 4):
    if last_str is not None:  
        max_otm_call, atm_call, max_otm_num, atm_num, dirc = last_str.split('_')  # 用之前开多仓的合约行权价和数量
        max_otm_call = float(max_otm_call)
        atm_call = float(atm_call)
        max_otm_num = float(max_otm_num)
        atm_num = float(atm_num)
        dirc = float(dirc)

        # check 有没有换合约
        last_max_otm_inst = last_option_current_date.loc[last_option_current_date['strike'] == float(max_otm_call), 'inst'].values[0]   # 老合约名称
        last_atm_inst =  last_option_current_date.loc[last_option_current_date['strike'] == float(atm_call), 'inst'].values[0]          # 老合约名称
        max_otm_inst = option_current_date.loc[option_current_date['strike'] == float(max_otm_call), 'inst'].values[0]                  # 当日合约名称
        atm_inst =  option_current_date.loc[option_current_date['strike'] == float(atm_call), 'inst'].values[0]                         # 当日合约名称
        # 如果移仓换月
        if (last_max_otm_inst != max_otm_inst)|(last_atm_inst != atm_inst):      
            path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_510300.XSHG_call_info_df.csv'
            option_traded = pd.read_csv(path, index_col=0)
            
            last_max_otm_symbol = option_traded.loc[option_traded['order_book_id']==int(last_max_otm_inst), 'symbol'].values[0]
            last_atm_symbol = option_traded.loc[option_traded['order_book_id']==int(last_atm_inst), 'symbol'].values[0]
            
            last_max_otm_sub_path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_{last_max_otm_inst}_{last_max_otm_symbol}_greeksSeries.csv'
            last_atm_sub_path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_{last_atm_inst}_{last_atm_symbol}_greeksSeries.csv'
            last_inst_max_otm_df = pd.read_csv(last_max_otm_sub_path, index_col=0)
            last_inst_max_otm_open = last_inst_max_otm_df.loc[open_minTime,'option_price']
            last_inst_atm_df = pd.read_csv(last_atm_sub_path, index_col=0)
            last_inst_atm_open = last_inst_atm_df.loc[open_minTime,'option_price']
            # print('移仓换月了 ',last_str, date, last_max_otm_inst, 'last_inst_max_otm_open',last_inst_max_otm_open, 'last_atm_inst',last_atm_inst,'last_inst_atm_open', last_inst_atm_open)

            if dirc == -1:
                max_otm_margin = max_otm_num*ETF_pre_close*last_contract_multiplier*0.12             # short max otm  
                atm_margin = last_inst_atm_open*atm_num*last_contract_multiplier                             # long atm      
            elif dirc == 1:
                max_otm_margin = max_otm_num*last_inst_max_otm_open*last_contract_multiplier         # long max otm  
                atm_margin = ETF_pre_close*atm_num*last_contract_multiplier*0.12                             # short atm  
        
            VSI_opencost = -dirc*(max_otm_num*last_inst_max_otm_open-atm_num*last_inst_atm_open)*last_contract_multiplier # 期权历史持仓价值
            # print("移仓换月了  get margin back  date", date, 'close_opencost',VSI_opencost, last_str)
            # print(date, 'last_inst_max_otm_open',last_inst_max_otm_open,'last_inst_atm_open', last_inst_atm_open,'last_contract_multiplier',last_contract_multiplier)
        # 没有移仓换月  
        else:  
            max_otm_open = float(option_current_date.loc[option_current_date['strike'] == float(max_otm_call), 'open'].values[0])
            atm_open = float(option_current_date.loc[option_current_date['strike'] == float(atm_call), 'open'].values[0])

            if dirc == -1:
                max_otm_margin = max_otm_num*ETF_pre_close*contract_multiplier*0.12            # short max otm  
                atm_margin = atm_open*atm_num*contract_multiplier                                      # long atm      
            elif dirc == 1:
                max_otm_margin = max_otm_num*max_otm_open*contract_multiplier                  # long max otm  
                atm_margin = ETF_pre_close*atm_num*contract_multiplier*0.12                            # short atm  
            VSI_opencost = -dirc*(max_otm_num*max_otm_open-atm_num*atm_open)*contract_multiplier # 期权历史持仓价值
            # print("没有移仓换月 正常平仓 in margin back  date", date, 'close_opencost',VSI_opencost, last_str)
            # print(last_str, date, 'max_otm_open',max_otm_open,'atm_open', atm_open,'contract_multiplier',contract_multiplier)

        total_margin = -(max_otm_margin + atm_margin)   # 无论是否移仓换月，都要平旧仓，保证金返还，增加cash
        VSI_commission = (atm_num+max_otm_num)*option_commission
        # print('VSI_commission in margin back ', VSI_commission)
        
    return total_margin, VSI_opencost, VSI_commission


# 以收盘价 计算持仓margin, 持仓value
def get_margin_value_by_close(last_str, ETF_pre_close, option_current_date, contract_multiplier): 
    max_otm_call, atm_call, max_otm_num, atm_num, dirc = last_str.split('_')# 用之前开多仓的 strikes, num
    max_otm_call = float(max_otm_call)
    atm_call = float(atm_call)
    max_otm_num = float(max_otm_num)
    atm_num = float(atm_num)
    dirc = float(dirc)
    
    max_otm_close = float(option_current_date.loc[option_current_date['strike'] == float(max_otm_call), 'close'].values[0])      # 当月合约的价格
    atm_close = float(option_current_date.loc[option_current_date['strike'] == float(atm_call), 'close'].values[0])              # 当月合约的价格


    if dirc == -1:
        max_otm_margin = max_otm_num*ETF_pre_close*contract_multiplier*0.12    # short max otm  last_max_otm_pre_close
        atm_margin = atm_close*atm_num*contract_multiplier                             # long atm  atm_open
            
    elif dirc == 1:
        max_otm_margin = max_otm_num*max_otm_close*contract_multiplier          # long max otm  max_otm_open
        atm_margin = ETF_pre_close*atm_num*contract_multiplier*0.12                     # short atm  last_atm_pre_close

    total_margin = max_otm_margin + atm_margin   # 无论买卖期权都交保证金, 不区分符号
    VSI_call_value = contract_multiplier*dirc*(max_otm_num*max_otm_close - atm_num*atm_close) 
    
    return total_margin, VSI_call_value




# 根据 rollover status 简化判断
def get_positions(factor_df, option_iv, option_current_month, option_commission:int = 3, otm_level:str = 'otm1_call', open_minTime:int = 4, risk_expo:float = 0.2, cash:int = 1000000):
    factor_df.reset_index(drop=True, inplace=True)
   
    strike_deltaRate_posRate_list = []
    strike_deltaRate_posRate = None
    strike_deltaRate_posRate_list.append(strike_deltaRate_posRate)
    VSI_call_opencost = 0
    VSI_call_commission = 0
    factor_df['VSI_call_opencost'] = 0
    factor_df['VSI_call_commission'] = 0


    VSI_close_value_list = []
    VSI_close_value = 0
    VSI_close_value_list.append(VSI_close_value)

    # 记录开/收盘的delta暴露
    far_cash_delta_list = []
    far_cash_delta = 0
    far_cash_delta_list.append(far_cash_delta)


    total_margin_list = []
    total_margin = 0
    total_margin_list.append(total_margin)


    # cash = 1000000  # 可用资金 初始为100万
    cash_list = []
    cash_list.append(cash)
    add_cash = 0    # 资金不足追加的保证金
    add_cash_list = []
    add_cash_list.append(add_cash)

    

    for i in range(1,factor_df.shape[0]):
        date = str(factor_df.loc[i, 'date'])[:10]
        last_date = str(factor_df.loc[i-1, 'date'])[:10]
        
        option_current_date = option_current_month.loc[option_current_month['time'] == date]
        last_option_current_date =  option_current_month.loc[option_current_month['time'] == last_date]
        ETF_open = factor_df.loc[i, 'ETF_open']
        ETF_close = factor_df.loc[i, 'ETF_close']
        ETF_pre_close = factor_df.loc[i-1, 'ETF_close']
        new_max_otm_call = option_current_date[otm_level].values[0]
        new_atm_call = option_current_date['atm_call'].values[0]

        # 当月合约的价格
        new_max_otm_open = option_current_date.loc[option_current_date['strike'] == float(new_max_otm_call), 'open'].values[0]     
        new_atm_open = option_current_date.loc[option_current_date['strike'] == float(new_atm_call), 'open'].values[0]  


        # 当月合约的希腊字母, 用于计算实时的组合希腊字母
        new_max_otm_delta = option_current_date.loc[option_current_date['strike'] == float(new_max_otm_call), 'open_delta'].values[0]
        new_atm_delta = option_current_date.loc[option_current_date['strike'] == float(new_atm_call), 'open_delta'].values[0]
        # new_max_otm_gamma = option_current_date.loc[option_current_date['strike'] == float(new_max_otm_call), 'open_gamma'].values[0]
        # new_atm_gamma = option_current_date.loc[option_current_date['strike'] == float(new_atm_call), 'open_gamma'].values[0]
        # new_max_otm_vega = option_current_date.loc[option_current_date['strike'] == float(new_max_otm_call), 'open_vega'].values[0]
        # new_atm_vega = option_current_date.loc[option_current_date['strike'] == float(new_atm_call), 'open_vega'].values[0]

        # 远月合约的希腊字母delta，用于确定开仓比例??   因为交易的是近月， 还是按照近月来开，在open_info里面有体现
        option_oneday =  option_iv.loc[option_iv['time'] == date]
        last_option_oneday =  option_iv.loc[option_iv['time'] == last_date]
        far_max_otm_delta = option_oneday.loc[option_oneday['strike'] == float(new_max_otm_call), 'open_delta'].values[0]          
        far_atm_delta = option_oneday.loc[option_oneday['strike'] == float(new_atm_call), 'open_delta'].values[0]   

        # 当月合约的合约乘数
        contract_multiplier = option_current_date.loc[option_current_date['strike'] == float(new_atm_call), 'contract_multiplier'].values[0] # 同一天不同虚实值的合约乘数相同,除了新上市的otm4,平值一般移动1-2档,不会移动四档
        last_contract_multiplier = last_option_current_date.loc[last_option_current_date['strike'] == float(new_atm_call), 'contract_multiplier'].values[0]

        last_str = strike_deltaRate_posRate_list[-1]    
        # 无论是否移仓换月，都要考虑的当前时刻的信号情况，只是需要移仓换月的处理上更麻烦
        #  open long， open short 不面临移仓换月，因为都没有持仓。 

        #  open short VSI = short otm + long atm
        if (factor_df.loc[i, 'final_signal'] == -1)&(factor_df.loc[i,'final_direction'] == -1):  # 开空仓
            direction = -1
            # print('开空仓', date)
            max_otm_num, atm_num, total_margin, new_cash_delta, far_cash_delta = get_open_infos(far_atm_delta, far_max_otm_delta, direction, ETF_open, ETF_pre_close, new_atm_open, new_max_otm_open, contract_multiplier, new_atm_delta, new_max_otm_delta, init_cash = 1000000, risk_expo = risk_expo)
            open_margin = total_margin

            cash -= total_margin  # total_margin 建仓无论开多开空期权都交保证金
            # factor_df.loc[i,'new_cash_delta'] = new_cash_delta
            

            strike_deltaRate_posRate = f'{new_max_otm_call}_{new_atm_call}_{max_otm_num}_{atm_num}_{direction}'

            # VSI_call_opencost >0 净开仓成本. 开空仓：卖max_otm(cost<0), 买atm(cost>0),只需记录每次新开仓的期权价格。 
            VSI_call_value = contract_multiplier*direction*(max_otm_num*new_max_otm_open - atm_num*new_atm_open)  # 注意以下,这里只包含(平仓后)初次开仓,new_pos==累计pos
            VSI_call_opencost = VSI_call_value 
            VSI_call_commission = option_commission*(max_otm_num+atm_num) 

            

        #  close_short VSI = long otm + short atm  VSI回到50分位数附近,且组合为净空头头寸
        elif ((factor_df.loc[i, 'final_signal'] == 2)&(factor_df.loc[i,'final_direction'] == 1))|((factor_df.loc[i, 'final_signal'] == 1)&(factor_df.loc[i,'final_direction'] == 0)):   # 平空仓 或有开多仓
            # 无论是否低于25分位数,都要先平空仓
            direction = 1
            total_margin, close_short_VSI_opencost, close_short_VSI_commission = get_margin_back(date, ETF_pre_close, last_str, option_current_date, last_option_current_date, contract_multiplier, last_contract_multiplier, option_commission = 3, open_minTime = open_minTime)
            
            if total_margin < 0:
                cash -= total_margin  # 平仓返回保证金  实时到账
            else:
                print('error! 平空仓 或有开多仓 total_margin', total_margin, date)  
            
            
            if (factor_df.loc[i, 'final_signal'] == 2): # 平空仓 + 开多仓
                print(f'final_signal = 2 平空 + 开多 {date}')
                max_otm_num, atm_num, total_margin, new_cash_delta, far_cash_delta = get_open_infos(far_atm_delta, far_max_otm_delta, direction, ETF_open, ETF_pre_close, new_atm_open, new_max_otm_open, contract_multiplier, new_atm_delta, new_max_otm_delta, init_cash = 1000000, risk_expo = risk_expo)
                open_margin = total_margin

                cash -= total_margin  # total_margin 建仓无论开多开空期权都交保证金
                # factor_df.loc[i,'new_cash_delta'] = new_cash_delta

                strike_deltaRate_posRate = f'{new_max_otm_call}_{new_atm_call}_{max_otm_num}_{atm_num}_{direction}'

                # VSI_call_opencost >0 净开仓成本. 开多仓：买max_otm(cost>0), 卖atm(cost<0),只需记录每次新开仓的期权价格。 
                VSI_call_value = contract_multiplier*direction*(max_otm_num*new_max_otm_open - atm_num*new_atm_open)

                # 无论有没有移仓换月，都有close_long_VSI_opencost, close_short_VSI_commission, 不同情况数值不同
                VSI_call_opencost = VSI_call_value + close_short_VSI_opencost                   # 新开仓成本+平空仓成本（移仓换月成本）
                VSI_call_commission = option_commission*(max_otm_num+atm_num) + close_short_VSI_commission  # 新开仓成本+平空仓手续费（移仓换月手续费）
                
            else:
                # 平仓的处理
                strike_deltaRate_posRate = None
                open_margin = 0
                # new_cash_delta = 0
                far_cash_delta = 0
                # factor_df.loc[i,'new_cash_delta'] = new_cash_delta

                # 无论有没有移仓换月，都有close_long_VSI_opencost, close_short_VSI_commission, 不同情况数值不同
                VSI_call_opencost= close_short_VSI_opencost   # 期权历史持仓价值
                VSI_call_commission = close_short_VSI_commission 
                
        

        #  open_long VSI = long otm + short atm
        elif  (factor_df.loc[i, 'final_signal'] == 1)&(factor_df.loc[i,'final_direction'] == 1):   # 开多仓
            direction = 1
            # print('开多仓', date)
            max_otm_num, atm_num, total_margin, new_cash_delta, far_cash_delta = get_open_infos(far_atm_delta, far_max_otm_delta, direction, ETF_open, ETF_pre_close, new_atm_open, new_max_otm_open, contract_multiplier, new_atm_delta, new_max_otm_delta, init_cash = 1000000, risk_expo = risk_expo)
            open_margin = total_margin

            cash -= total_margin  # total_margin 建仓无论开多开空期权都交保证金
            # factor_df.loc[i,'new_cash_delta'] = new_cash_delta

            strike_deltaRate_posRate = f'{new_max_otm_call}_{new_atm_call}_{max_otm_num}_{atm_num}_{direction}'

            # VSI_call_opencost >0 净开仓成本. 开多仓：买max_otm(cost>0), 卖atm(cost<0),只需记录每次新开仓的期权价格。new_atm_pos已含有方向
            VSI_call_value = contract_multiplier*direction*(max_otm_num*new_max_otm_open - atm_num*new_atm_open)
            VSI_call_opencost = VSI_call_value
            VSI_call_commission = option_commission*(max_otm_num+atm_num) 



        #  close_long VSI = short otm + long atm
        elif ((factor_df.loc[i, 'final_signal'] == -1)&(factor_df.loc[i,'final_direction'] == 0))|((factor_df.loc[i, 'final_signal'] == -2)&(factor_df.loc[i,'final_direction'] == -1)):    # 平多仓 或有开空仓
            # 无论是否高于75分位数,都要先平多仓
            direction = -1
            total_margin, close_long_VSI_opencost, close_long_VSI_commission = get_margin_back(date, ETF_pre_close, last_str, option_current_date, last_option_current_date, contract_multiplier, last_contract_multiplier, option_commission = 3, open_minTime = open_minTime)
            
            
            if total_margin < 0:
                cash -= total_margin  # 平仓返回保证金  实时到账
            else:
                print('error! 平空仓 或有开多仓 total_margin', total_margin, date)  
            

            if (factor_df.loc[i, 'final_signal'] == -2): # 平多仓 + 开空仓
                print(f'final_signal = -2 平多 + 开空 {date}')
                max_otm_num, atm_num, total_margin, new_cash_delta, far_cash_delta = get_open_infos(far_atm_delta, far_max_otm_delta, direction, ETF_open, ETF_pre_close, new_atm_open, new_max_otm_open, contract_multiplier, new_atm_delta, new_max_otm_delta, init_cash = 1000000, risk_expo = risk_expo)
                open_margin = total_margin

                cash -= total_margin  # total_margin 建仓无论开多开空期权都交保证金
                # factor_df.loc[i,'new_cash_delta'] = new_cash_delta

                strike_deltaRate_posRate = f'{new_max_otm_call}_{new_atm_call}_{max_otm_num}_{atm_num}_{direction}'
                
                # VSI_call_opencost >0 净开仓成本. 开多仓：买max_otm(cost>0), 卖atm(cost<0),只需记录每次新开仓的期权价格。 
                VSI_call_value = contract_multiplier*direction*(max_otm_num*new_max_otm_open - atm_num*new_atm_open)

                # 无论有没有移仓换月，都有close_long_VSI_opencost, close_short_VSI_commission, 不同情况数值不同
                VSI_call_opencost = VSI_call_value + close_long_VSI_opencost 
                VSI_call_commission = option_commission*(max_otm_num+atm_num) + close_long_VSI_commission 
               


    
            elif (factor_df.loc[i,'final_direction'] == 0):  # 平仓的处理            
                strike_deltaRate_posRate = None
                open_margin = 0
                # new_cash_delta = 0
                far_cash_delta = 0
                # factor_df.loc[i,'new_cash_delta'] = new_cash_delta

                # 无论有没有移仓换月，都有close_long_VSI_opencost, close_short_VSI_commission, 不同情况数值不同
                VSI_call_opencost = close_long_VSI_opencost  # 期权历史持仓价值
                VSI_call_commission = close_long_VSI_commission 


        
        elif (factor_df.loc[i, 'final_signal'] == 0)&(factor_df.loc[i,'final_direction'] != 0): # 持仓信号不变的时刻
            max_otm_call, atm_call, max_otm_num, atm_num, dirc = last_str.split('_')  # 用之前开仓的 strikes, option num 
            max_otm_call = float(max_otm_call)
            atm_call = float(atm_call)
            max_otm_num = float(max_otm_num)
            atm_num = float(atm_num)
            dirc = float(dirc)

            # check 有没有换合约
            last_max_otm_inst = last_option_current_date.loc[last_option_current_date['strike'] == (max_otm_call), 'inst'].values[0]   # 老合约名称
            last_atm_inst =  last_option_current_date.loc[last_option_current_date['strike'] == (atm_call), 'inst'].values[0]          # 老合约名称
            max_otm_inst = option_current_date.loc[option_current_date['strike'] == (max_otm_call), 'inst'].values[0]                  # 当日合约名称
            atm_inst =  option_current_date.loc[option_current_date['strike'] == (atm_call), 'inst'].values[0]                         # 当日合约名称


            # 如果移仓换月
            if (last_max_otm_inst != max_otm_inst)|(last_atm_inst != atm_inst):  
                path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_510300.XSHG_call_info_df.csv'
                option_traded = pd.read_csv(path, index_col=0)
                
                last_max_otm_symbol = option_traded.loc[option_traded['order_book_id']==int(last_max_otm_inst), 'symbol'].values[0]
                last_atm_symbol = option_traded.loc[option_traded['order_book_id']==int(last_atm_inst), 'symbol'].values[0]
                
                last_max_otm_sub_path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_{last_max_otm_inst}_{last_max_otm_symbol}_greeksSeries.csv'
                last_atm_sub_path = f'/home/zhangyan/option_skew/data-open-20220101_20240731/{date}/{date}_{last_atm_inst}_{last_atm_symbol}_greeksSeries.csv'
                last_inst_max_otm_df = pd.read_csv(last_max_otm_sub_path, index_col=0)
                last_inst_max_otm_open = last_inst_max_otm_df.loc[open_minTime,'option_price']
                last_inst_atm_df = pd.read_csv(last_atm_sub_path, index_col=0)
                last_inst_atm_open = last_inst_atm_df.loc[open_minTime,'option_price']
               
                max_otm_open = option_current_date.loc[option_current_date['strike'] == (max_otm_call), 'open'].values[0]               # 新合约当日对应的价格
                atm_open = option_current_date.loc[option_current_date['strike'] == (atm_call), 'open'].values[0]                       # 新合约当日对应的价格

                if dirc == -1:
                    last_max_otm_margin = max_otm_num*ETF_pre_close*last_contract_multiplier*0.12             # short max otm  
                    last_atm_margin = last_inst_atm_open*atm_num*last_contract_multiplier                             # long atm  
                    max_otm_margin = max_otm_num*ETF_pre_close*contract_multiplier*0.12             # short max otm  
                    atm_margin = atm_open*atm_num*contract_multiplier                                       # long atm    

                elif dirc == 1:
                    last_max_otm_margin = max_otm_num*last_inst_max_otm_open*last_contract_multiplier         # long max otm  
                    last_atm_margin = ETF_pre_close*atm_num*last_contract_multiplier*0.12                             # short atm  
                    max_otm_margin = max_otm_num*max_otm_open*contract_multiplier                    # long max otm  
                    atm_margin = ETF_pre_close*atm_num*contract_multiplier*0.12                              # short atm 
            
                close_margin = -(last_max_otm_margin + last_atm_margin)   # 无论是否移仓换月，都要平旧仓，保证金返还，增加cash
                if close_margin < 0:
                    cash -= close_margin  # 平仓返回保证金  实时到账
                else:
                    print(' Error! 信号不变的时刻 string', date, last_str, factor_df.loc[i, 'final_signal'], factor_df.loc[i,'final_direction'])  
                    break
                open_margin = (max_otm_margin + atm_margin) 
                cash -= open_margin


                # opencost是用来计算pnl的，不是费用，不必双向记录，分开两次思考
                close_VSI_opencost = -dirc*(max_otm_num*last_inst_max_otm_open - atm_num*last_inst_atm_open)*last_contract_multiplier # 期权历史持仓价值
                VSI_opencost = dirc*(max_otm_num*max_otm_open - atm_num*atm_open)*contract_multiplier + close_VSI_opencost# 期权历史持仓价值
                VSI_commission = 2*(max_otm_num+atm_num)*option_commission # 持仓手数不变，换合约，双倍手续费
                
                # print('信号不变+移仓换月!  now date:', date,'total_margin', total_margin, 'close_margin', close_margin, 'open_margin', open_margin)
                VSI_call_opencost = VSI_opencost   # 期权历史持仓价值
                VSI_call_commission = VSI_commission 
            else: # 没有进行移仓换月
                VSI_call_opencost =  0 # 期权历史持仓价值
                VSI_call_commission =  0

            
            # 当前只有open计算的greeks,且greeks记录的都是用open 开仓的strike可能与当前max otm ,atm不同，所以不能直接用new_greeks
            # 无论是否移仓换月，都要计算greeks
            max_otm_delta = option_oneday.loc[option_oneday['strike'] == (max_otm_call), 'open_delta'].values[0]
            atm_delta = option_oneday.loc[option_oneday['strike'] == (atm_call), 'open_delta'].values[0]

            far_cash_delta = dirc*(max_otm_delta*max_otm_num - atm_delta*atm_num)*ETF_open*contract_multiplier

            
            # print('持仓',date,last_str, new_cash_delta, max_otm_delta,max_otm_num, atm_delta,atm_num,ETF_open,contract_multiplier, '\n' )

            


        # T0 收盘后 结算保证金 进行风控
        
        last_str = strike_deltaRate_posRate  # 最新的strike_deltaRate_posRate
        if last_str is not None:
            total_margin_at_close, VSI_close_value = get_margin_value_by_close(last_str, ETF_pre_close, option_current_date, contract_multiplier)

            if cash < (total_margin_at_close - open_margin):
                add_cash = (total_margin_at_close - open_margin) + cash
                total_margin += add_cash
                print('充值！！！', date, add_cash)
        else:
            VSI_close_value = 0

        add_cash_list.append(add_cash)
        add_cash = 0

        factor_df.loc[i,'VSI_call_opencost'] = VSI_call_opencost 
        factor_df.loc[i,'VSI_call_commission'] = VSI_call_commission 
        VSI_call_opencost = 0
        VSI_call_commission = 0
 
        # 记录开仓的cash delta 和 收盘的cash delta
        far_cash_delta_list.append(far_cash_delta)
        far_cash_delta = 0
       
  
        total_margin_list.append(total_margin)
        VSI_close_value_list.append(VSI_close_value)
        cash_list.append(cash)
        VSI_close_value = 0
        
        strike_deltaRate_posRate_list.append(strike_deltaRate_posRate)



    factor_df['K_rate_pos_dirc'] = strike_deltaRate_posRate_list
    factor_df['far_cash_delta'] = far_cash_delta_list
    factor_df['total_margin'] = total_margin_list


    # VSI 的持仓价值
    factor_df['VSI_close_value'] = VSI_close_value_list

    # call options pnl
    factor_df['VSI_call_pnl'] = factor_df['VSI_close_value'].diff() - factor_df['VSI_call_opencost']
    factor_df['net_VSI_call_pnl'] = factor_df['VSI_call_pnl'] - factor_df['VSI_call_commission']


    # accum pnls
    factor_df['accum_call_pnl'] = factor_df['net_VSI_call_pnl'].cumsum()  # 有手续费
    # factor_df['accum_call_pnl'] = factor_df['VSI_call_pnl'].cumsum() # 没有手续费

    # accum costs
    factor_df['accum_call_commission'] = factor_df['VSI_call_commission'].cumsum()

    factor_df['cash'] = cash_list


    # factor_df['diff_call_pnl'] = factor_df['accum_call_pnl'].diff()
    # print('max gap',factor_df['diff_call_pnl'].dropna().sort_values()[-5:])
    # print('min gap',factor_df['diff_call_pnl'].dropna().sort_values()[:5])

    print('noadjust_22-24最终结果',factor_df.tail())
    factor_df.to_csv(f'noadjust_22-24_{otm_level[:4]}_atm_pnl_df.csv')
    # print('不合格的位置',  factor_df.loc[abs(factor_df['far_cash_delta'])>200000,['date','final_signal','final_direction','far_cash_delta','accum_call_pnl']])
    return factor_df
            


def get_max_drawdown(factor_df:pd.DataFrame, init_cash = 1000000):
    factor_df = factor_df[(factor_df['accum_call_pnl'] != 0)&(factor_df['accum_call_pnl'].notna())].copy() # .loc[:'2023-12-22']
    # 计算年化收益率水平
    factor_df['daily_return_rate'] =  factor_df['net_VSI_call_pnl']/(init_cash + factor_df['accum_call_pnl'].shift(1))
    # print(factor_df[:25], factor_df['daily_return_rate'], factor_df[-25:])

    avg_daily_return = np.mean(factor_df['daily_return_rate'])
    # print(avg_daily_return)
    annual_return = (1 + avg_daily_return)**252 - 1

    # 计算最大回撤水平
    peak = np.maximum.accumulate(factor_df['accum_call_pnl']) + init_cash
    drawdown = (init_cash + factor_df['accum_call_pnl'] - peak)/peak
    max_drawdown = np.min(drawdown)
    print('annual_return', annual_return)
    print('max_drawdown', max_drawdown)
    factor_df.to_csv('noadjust_22-24_otm1_atm_drawdown_return.csv')
    return annual_return, max_drawdown
            


# ## no costs plot
def get_VSI_nocosts_pnl_plot(plot_df, otm_level:str = 'otm1_call'):  
    plt.figure(figsize = (30,8))
    plt.plot(plot_df.index, plot_df['accum_call_pnl'], color='red', label = 'accumulated call pnl')

    plt.xticks(rotation=45)
    plt.xlabel('days')
    plt.ylabel('pnls')
    plt.title('accumulated pnls')
    plt.legend()
    plt.savefig(f'noadjust_22-24_{otm_level[:4]}_atm_pnls_nocost.png')
    plt.show()                      


## costs plot
def get_VSI_costs_pnl_plot(plot_df, otm_level:str = 'otm1_call'):  
    plt.figure(figsize = (30,8))
    # print(plot_df.columns)

    Y23_begin = plot_df[plot_df['date'] == '2023-01-03'].index
    Y23_end = plot_df[plot_df['date'] == '2023-12-29'].index
    plt.subplot(1,2,1)
    plt.plot(plot_df.index, plot_df['accum_call_pnl'], color='red', label = 'accumulated call pnl')
    plt.axvline(x = Y23_begin,color = 'blue', linestyle = '-.')
    plt.axvline(x = Y23_end,color = 'blue', linestyle = '--')

    plt.xticks(rotation=45)
    plt.xlabel('days')
    plt.ylabel('pnls')
    plt.title('accumulated pnls')
    plt.legend()


    plt.subplot(1,2,2)
    # plt.plot(plot_df.index, plot_df['accum_total_commission'], color='red', label = 'accumulated total costs')
    plt.plot(plot_df.index, plot_df['accum_call_commission'], color='cyan', label = 'accumulated call cost')
    # plt.plot(plot_df.index, plot_df['accum_ETF_commission'], color='orange', label = 'accumulated ETF cost')
    plt.xticks(rotation=45)
    plt.xlabel('days')
    plt.ylabel('costs')
    plt.title('accumulated costs')
    plt.legend()
    plt.savefig(f'noadjust_22-24_{otm_level[:4]}_atm_pnls_costs.png')

    plt.show()              


import seaborn as sns

def get_cash_delta_plot(plot_df, otm_level:str = 'otm1_call'):
    plot_df = plot_df[~plot_df['far_cash_delta'].isna()].copy()
    plot_df.reset_index(drop=True, inplace=True)
    plt.figure(figsize = (10,6))
    
    # plt.xticks(rotation=45)
                    
    # plt.hist(plot_df['new_cash_delta'], bins = 50, edgecolor = 'black')
    plt.plot(plot_df['far_cash_delta'])
    plt.xlabel('days')
    plt.ylabel('cash_delta', color= 'tab:blue')
     
    plt.title('Cash Delta Distribution')  
    plt.show()
    plt.savefig(f'noadjust_22-24_{otm_level[:4]}_atm_cash_delta_dist.png')





if __name__ =="__main__":
    target_folder = '/home/zhangyan/option_skew/data-open-20220101_20240731'
    start_date =  '2022-02-11'#'2022-02-11'
    end_date = '2024-01-10' # '2023-12-31'#'2024-01-10'
    trading_date_list = get_rollingDates(target_folder)
    ETF_daily = get_parallel_ETF_Data(trading_date_list)
    ETF_daily = get_strikes(ETF_daily, start_date = start_date, end_date = end_date) 



    # 多进程
    option_iv = get_parallel_option_daily(ETF_daily, trading_date_list, signal_ttm = 45, open_minTime = 4, close_minTime = 229)  # signal_ttm = 45  # 得到 option iv & greeks 信号上移仓换月的处理（>=57）
    option_current_month_iv = get_parallel_option_current_month(ETF_daily, trading_date_list, min_ttm = 5, open_minTime = 4, close_minTime = 229)  # min_ttm = 5  # 得到 option iv & greeks : call_min  信号上移仓换月的处理（>=10）
    
    factor_df = get_timing_signal(option_current_month_iv, otm_level = 'otm1_call', timing_path = '/home/zhangyan/option_skew/timing_signal.csv',start_date=start_date, end_date = end_date)  # 计算距离到期日10天时，移仓换月的情况        
    

    skews_target_df = get_positions(factor_df = factor_df, option_iv = option_iv, option_current_month = option_current_month_iv, open_minTime = 4, risk_expo = 0.2)   
    get_VSI_costs_pnl_plot(skews_target_df, otm_level = 'otm1_call')
    get_max_drawdown(factor_df, init_cash = 1000000)
    # # # get_VSI_nocosts_pnl_plot(skews_target_df, otm_level = 'otm1_call')
    get_cash_delta_plot(skews_target_df, otm_level = 'otm1_call')


    # # start_date =   '2023-07-04'  #'2023-09-01'
    # # end_date =  '2023-07-13'  # '2023-10-23'
    # # neg_df = skews_target_df.loc[(skews_target_df['date']>=start_date)&(skews_target_df['date']<=end_date), ['date','final_signal','direction','far_cash_delta','new_cash_delta','accum_call_pnl','accum_call_commission']]
    # # # sorted_df = skews_target_df.dropna().sort_values(by='accum_call_pnl') 
    # # # print(sorted_df.iloc[-40:])
   
#!/usr/bin/env python
# coding: utf-8

# In[507]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Purpose

# This notebook provides a high level overview of some fundamental concepts within algorithmic trading. The idea is that implementation strengthens understanding of which makes further innovation possible.

# # Problem Statement

# ## Aim of Market Making

# To quote bids and offers consistent with the obligation to provide liquidity. This implies that the market maker must account for a percentage of the volume traded in an asset. Likewise, a key concern of the market maker is ensuring that inventory is available to fulfill this obligation and meet market demand across various market conditions. Modeling future expected volume, volatitilty, order arrival rates, and the limit order book are some means of managing inventory risks.
# 

# ## Aim of Execution Algorithms

# To execute trades in a security within a given timeframe toward the end of price improvement. This entails the tracking of some benchmark (i.e. VWAP, TWAP,etc) as a means of determining execution quality. Different approaches can be taken given market conditions such as AIM (aggressive in the money) and PIM (passive in the money) tactics. A key component of the implementation of execution algorithms is the modeling of the volume distribution.

# # Load Dataset

# ### 3 second snapshot data of e-Mini S&P 500 Futures

# In[3]:


data=pd.read_excel('ES 3sec Research Dataset.xlsx')


# In[4]:


data.head()


# # Volume Profile Analysis

# ## Distribution of Volume by Time of Day

# In[234]:


volume_data=data[['Date Time','Volume']]


# In[438]:


fig, axs = plt.subplots(figsize=(15, 8))

volume_data.groupby(volume_data["Date Time"].dt.hour)["Volume"].mean().plot(
    kind='bar', rot=0, ax=axs
)
plt.title('ESH23 Futures Trading Volume')
plt.xlabel("Hour of Trading Session")
plt.ylabel("Volume")
plt.legend(loc=0)
plt.tight_layout()


# # Feature Engineering 

# In[5]:


#adding average volume per window constraint as means of setting execution policy
data['Avg. Vol']=data['Volume'].rolling(window=30).mean()


# In[7]:


data.fillna(0,inplace=True)


# In[83]:


#adding moving average as proxy for benchmark(i.e. VWAP, TWAP, etc)
data['MA']=round(data['Last'].rolling(window=3).mean(),2)


# In[84]:


data.fillna(0,inplace=True)


# In[86]:


data.head()


# # Mock Execution Algorithm Implementation

# # Synthetic Order Queue

# To begin we will create a means for generating synthetic orders. The key considerations are the side, priority, and size. These orders will be generated randomly and our execution strategy will use the priority and the corresponding volume profile prediction to determine how best to execute the order.

# In[53]:


class OrderQueue:
    
    def create_orders(self,num_orders=np.random.randint(30)):
        
        self.orders_in_queue=[]
        
        for pending in range(0,num_orders):
            side=np.random.choice(['Buy','Sell'])
            lot_size=np.random.choice(np.arange(10,1000))
            urgency=np.random.choice(['High','Medium','Low'])

            order={'order_id':np.random.choice(np.arange(1000,2000)),
                    'side':side,
                   'lot_size':lot_size,
                    'urgency':urgency}
            
            self.orders_in_queue.append(order) 


# ## Testing Order Queue Implementation

# In[54]:


pending_orders=OrderQueue()


# In[55]:


pending_orders.create_orders()


# In[524]:


pending_orders.orders_in_queue


# # Establishing Volume Profile Projections

# We want our algorithm to execute orders during periods of high liquidity. This is to avoid market impact, signaling risk, and achieve price improvement. For this implementation we bin our volume data and compute limits of liquidity. During the auction, buyers and sellers tend to identify areas where they are most willing to transact business. These areas tend to be high volume areas. These are the areas that we want to identify and execute our orders in as they increase the probability of price improvement and minimize implementation shortfalls.

# In this implementation, we first identify the prices in which buyers and sellers deem are fair value. We then identify the prices that constitute the range in which 70% of the volume was traded. In some implentations, volume is assessed on a time basis. Here, we assess volume by price as volume and price are independent of time and we would like to know the areas most optimal for executing orders and not necessarily the time of day to execute orders.

# These price limits will be used later to determine the range in which we should execute orders based on liquidity. Our volume profile analysis will return an upper and lower price limit consistent with the area where the majority of transactions took place.

# In[344]:


class VolumeProfile:
    
    def __init__(self,data,lookback):
        self.lookback=lookback
        self.data=data
        
    def create_bins(self):
        '''Creates bins for volume profiles.'''
        #create snapshot bins based on the lookback period
        self.bins=[]
        self.num_bins=len(self.data)//self.lookback
        
        for index in range(0,self.num_bins):
            
            #select subset
            subset=self.data[self.lookback*index:self.lookback+(index*self.lookback)]
            
            #append to bins
            self.bins.append(subset)
            
        
        return
    
    def get_num_bins(self):
        '''helper method that is used as input into generate of order queue.'''
        return self.num_bins
    
    def create_profiles(self):
        '''Create volume profiles for each snapshot interval.'''
        
        #list of dataframes that capture volume traded at each prices over interval
        self.volume_profiles=[]
        
        #iterate over bins
        for subset in self.bins:
            
            #create unique set of prices and volumes trade at price
            current_profile={}
            
            unique_prices=np.unique(subset['Last'])
            
            sorted_prices=sorted(unique_prices)
            
            #grab all volume traded at price over lookback interval
            for price in sorted_prices:
                subset_of_current_price=subset[subset['Last']==price]
                
                #add price and cumulative volume at price to profile
                current_profile[price]=subset_of_current_price['Volume'].sum()
                
            #adding current profile to volume profiles list
            self.volume_profiles.append(current_profile)
            
        return
    
    
    def set_limits(self):
        
        self.limits=[]
        
        for profile in self.volume_profiles:
            if len(profile) >= 10:
                #convert dict to dataframe
                #note: after this transformation, price is index and 0 col name is the volume
                df=pd.DataFrame.from_dict(profile,orient='index')
                
                try:
                    #identify percentile limits(i.e. where 70% of volume was traded, 15th and 85th percentiles)
                    #determine prices at percentiles
                    upper_vol_limit=np.percentile(df[0],85)
                    lower_vol_limit=np.percentile(df[0],15)
                    
                    #get prices that correspond to volume thresholds
                    upper_price_limit=max(df.loc[df[0]<=upper_vol_limit].index)
                    lower_price_limit=min(df.loc[df[0]>=lower_vol_limit].index)
                    
                    #append limits
                    self.limits.append((round(upper_price_limit,2),round(lower_price_limit,2)))
                
                except Exception as e:
                    print(profile)
                    print(e)
                    continue
            
        return 
            
        
    def generate_profiles(self):
        
        self.create_bins()
        self.create_profiles()
        self.set_limits()
        
        return 


# ## Testing Volume Profiles Creation

# In[320]:


profiles=VolumeProfile(data,lookback=1000)


# In[321]:


profiles.generate_profiles()


# In[322]:


profiles.bins


# In[323]:


profiles.limits


# # Determining Priority

# The priority of an order has a direct relationship to how it is executed. A POV algorithm is designed to take liquidity based on an order's level of urgency. The more urgent, the more aggressive, or larger the percentage of volume traded.

# In[329]:


'''Determines the POV based on level of urgency.'''
urgency={   
    
    'High':0.15,
    'Medium':0.075,
    'Low':0.25
}


# # Determining Trading Policy

# There are a variety of techniques for executing orders. Below for the mock execution implementation, we'll use aggressive and passive in the money policies in conjunction with a percent of volume algorithm. The POV algorithm, leveraging the projected liquidity from the volume profile analysis, seeks to account for a percentage of volume based on an order's level of urgency. AIM and PIM tactics are two approaches that determine how to execute orders when in-the-money. One common benchmark for determining in-the-money is VWAP. For this implementation we keep things simple and use a moving average over price.

# In[427]:


class Policy:
    
    def __init__(self,urgency_lookup):
        self.urgency_lookup=urgency_lookup
    
    def passive_in_money(self,data,order,start_index,current_volume_price_limits,urgency):
        last_timestamp=start_index
        
        for index in data[start_index:].index:
            executions=[]

            print(f"Bid Size: {data.iloc[index]['Bid Size']}| Bid Volume:{data.iloc[index]['Bid Volume']}| Mid: {data.iloc[index]['Last']}| Ask Size: {data.iloc[index]['Ask Size']}| Ask Volume:{data.iloc[index]['Ask Volume']}| Volume:{data.iloc[index]['Volume']}")
            #Long
            pov=self.urgency_lookup.get(urgency)

            #determine if trading in most liquid area based on volume profile
            if data.iloc[index]['Last'] <= current_volume_price_limits[0] or data.iloc[index]['Last'] >= current_volume_price_limits[1]:
                #take the offer (i.e. market order)
                quantity=int(round(min(order['lot_size'],pov*data.iloc[index]['Volume']),1))
                if quantity < 1:
                    quantity=1
                remainder=order['lot_size']-quantity
                trade_execution={'timestamp':data.iloc[index]['Date Time'],'strategy':'PIM','side':order['side'],"order_type":'Limit',"price":data.iloc[index]['Last'],"quantity":quantity}

                executions.append(trade_execution)

                if not remainder:
                    last_timestamp=index                       
                    break


                executions_df=pd.json_normalize(executions)

                avg_exec_price=executions_df['price'].mean()

                price_improvement=data.iloc[last_timestamp]['MA']-avg_exec_price

                trade_log={'order_id':order['order_id'],'executions':executions,'price_improvement':price_improvement}

                print('/n')
                print('Trade Log:')
                print(trade_log)
                print('/n')
                return trade_log,last_timestamp 
                
    
    def aggressive_in_money(self,data,order,start_index,current_volume_price_limits,urgency):
        last_timestamp=start_index
        
        for index in data[start_index:].index:
            executions=[]

            print(f"Bid Size: {data.iloc[index]['Bid Size']}| Bid Volume:{data.iloc[index]['Bid Volume']}| Mid: {data.iloc[index]['Last']}| Ask Size: {data.iloc[index]['Ask Size']}| Ask Volume:{data.iloc[index]['Ask Volume']}| Volume:{data.iloc[index]['Volume']}")
            
            #if in the money
            #Long
            if order['side']=='Buy' and data.iloc[index]['Last'] < data.iloc[index]['MA']:
                pov=self.urgency_lookup.get(urgency)
                
                #determine if trading in most liquid area based on volume profile
                if data.iloc[index]['Last'] <= current_volume_price_limits[0] or data.iloc[index]['Last'] >= current_volume_price_limits[1]:
                    #take the offer (i.e. market order)
                    quantity=int(round(min(order['lot_size'],pov*data.iloc[index]['Volume']),1))
                    if quantity < 1:
                        quantity=1
                    remainder=order['lot_size']-quantity
                    trade_execution={f'timestamp':data.iloc[index]['Date Time'],'strategy':'AIM','side':'Buy',"order_type":'Market',"price":data.iloc[index]['Last'],"quantity":quantity}

                    executions.append(trade_execution)

                    if not remainder:
                        last_timestamp=index                       
                        break


                    executions_df=pd.json_normalize(executions)

                    avg_exec_price=executions_df['price'].mean()

                    price_improvement=data.iloc[last_timestamp]['MA']-avg_exec_price

                    trade_log={'order_id':order['order_id'],'executions':executions,'price_improvement':price_improvement}

                    print('/n')
                    print('Trade Log:')
                    print(trade_log)
                    print('/n')
                    return trade_log,last_timestamp 

                else:#out of money long
                    #add liquidity( i.e. limit order)
                    #determine if trading in most liquid area based on volume profile
                    if data.iloc[index]['Last'] <= current_volume_price_limits[0] or data.iloc[index]['Last'] >= current_volume_price_limits[1]:
                        #take the offer (i.e. market order)
                        quantity=int(ropund(min(order['lot_size'],pov*data.iloc[index]['Volume']),1))
                        if quantity < 1:
                            quantity=1
                        remainder=order['lot_size']-quantity
                        trade_execution={f'timestamp':data.iloc[index]['Date Time'],'strategy':'AIM','side':'Buy',"order_type":'Market',"price":data.iloc[index]['Last'],"quantity":quantity}

                        executions.append(trade_execution)

                        if not remainder:
                            last_timestamp=index                       
                            break


                        executions_df=pd.json_normalize(executions)

                        avg_exec_price=executions_df['price'].mean()

                        price_improvement=data.iloc[last_timestamp]['MA']-avg_exec_price

                        trade_log={'order_id':order['order_id'],'executions':executions,'price_improvement':price_improvement}
                        print('/n')
                        print('Trade Log:')
                        print(trade_log)
                        print('/n')
                        
                        return trade_log,last_timestamp 

                    
            #Short in the money
            elif order['side']=='Sell' and data.iloc[index]['Last'] > data.iloc[index]['MA']:
                pov=self.urgency_lookup.get(urgency)
                
                #determine if trading in most liquid area based on volume profile
                if data.iloc[index]['Last'] <= current_volume_price_limits[0] or data.iloc[index]['Last'] >= current_volume_price_limits[1]:
                    #take the offer (i.e. market order)
                    quantity=int(round(min(order['lot_size'],pov*data.iloc[index]['Volume']),1))
                    if quantity < 1:
                        quantity=1
                    remainder=order['lot_size']-quantity
                    trade_execution={f'timestamp':data.iloc[index]['Date Time'],'strategy':'AIM','side':'Sell',"order_type":'Market',"price":data.iloc[index]['Last'],"quantity":quantity}

                    executions.append(trade_execution)

                    if not remainder:
                        last_timestamp=index                       
                        break


                    executions_df=pd.json_normalize(executions)

                    avg_exec_price=executions_df['price'].mean()

                    price_improvement=data.iloc[last_timestamp]['MA']-avg_exec_price

                    trade_log={'order_id':order['order_id'],'executions':executions,'price_improvement':price_improvement}

                    print('/n')
                    print('Trade Log:')
                    print(trade_log)
                    print('/n')
                    return trade_log,last_timestamp 

                #short out of the money
                else:
                    #add liquidity( i.e. limit order)
                    #determine if trading in most liquid area based on volume profile
                    if data.iloc[index]['Last'] <= current_volume_price_limits[0] or data.iloc[index]['Last'] >= current_volume_price_limits[1]:
                        #take the offer (i.e. market order)
                        quantity=int(round(min(order['lot_size'],pov*data.iloc[index]['Volume']),1))
                        if quantity < 1:
                            quantity=1
                        remainder=order['lot_size']-quantity
                        trade_execution={f'timestamp':data.iloc[index]['Date Time'],'strategy':'AIM','side':'Sell',"order_type":'Limit',"price":data.iloc[index]['Last'],"quantity":quantity}

                        executions.append(trade_execution)

                        if not remainder:
                            last_timestamp=index                       
                            break


                        executions_df=pd.json_normalize(executions)

                        avg_exec_price=executions_df['price'].mean()

                        price_improvement=data.iloc[last_timestamp]['MA']-avg_exec_price
                        
                        
                        trade_log={'order_id':order['order_id'],'executions':executions,'price_improvement':price_improvement}
                        print('/n')
                        print('Trade Log:')
                        print(trade_log)
                        print('/n')
                        return trade_log,last_timestamp 

                
                

                


# # Execution Strategy Implementation

# At this point, we're ready to create our execution strategy implementation. The strategy will follow a Percent of Volume mandate that is dynamic based on the whether or not we're trading in the money. This implementation will assess the urgency of the order and if that urgency is high, will implement the AIM policy, otherwise, it will implement the PIM policy. 

# The orders will be generated randomly using the OrderQueue class. Then random indicies will be selected as means of simulating the arrival of orders. The execution strategy will assess the urgency and execute the appropriate policy. The policy will use the Volume Profile limits to determine where liquidity is and if trades should be executed. If we are not trading in an areas of high liquidity, the strategy will not trade.

# Once the level of urgency and liquidity level are defined, the strategy-policy will monitor our benchmark, here which is an SMA, but typically VWAP, TWAP, etc. to determine if we're trading in the money. The AIM will be aggressive (i.e. submit market orders) in the money and passive(i.e. limit orders) out of the money. The PIM will be passive at all times.

# In[428]:


class ExecutionStrategy:
    
    def __init__(self,data,urgency_pov):
        #generate volume profiles
        self.volume_profiles=VolumeProfile(data,lookback=1000)
        self.volume_profiles.generate_profiles()
        self.volume_limits=self.volume_profiles.limits
        
        
        #create order queue
        self.order_queue=OrderQueue()
        self.order_queue.create_orders()
        self.pending_orders=self.order_queue.orders_in_queue
        self.order_indicies=[np.random.randint(len(self.pending_orders)) for num in range(0,round(int(np.log(len(self.pending_orders))),2)) ]
        
        self.urgency_pov=urgency_pov
        
        self.executions=[]
     
    def execute_orders(self):
        #variable to track transition between executions
        last_timestamp=0
        order_recieved=0

        #iterate over market data
        for timestamp in range(last_timestamp,len(data)):
            #if last_timestamp >= timestamp:
            print(f"Bid Size: {data.iloc[timestamp]['Bid Size']}| Bid Volume:{data.iloc[timestamp]['Bid Volume']}| Mid: {data.iloc[timestamp]['Last']}| Ask Size: {data.iloc[timestamp]['Ask Size']}| Ask Volume:{data.iloc[timestamp]['Ask Volume']}| Volume:{data.iloc[timestamp]['Volume']}")

            #instantiating trading policy
            policy=Policy(self.urgency_pov)

            if timestamp in self.order_indicies:
                #get order
                current_order=self.pending_orders[timestamp]
                print('---------------------')
                print('----Order Received---')
                print(current_order)
                print('---------------------')

                #get urgency
                current_priority=current_order.get('urgency')

                #start_index is the timestamp following when the order request was made
                start_index=timestamp+1

                #limits from volume profile that determine most liquid area to execute orders
                current_volume_profile_limits=self.volume_limits[timestamp]

                #determine execution policy
                if current_priority=='High':
                    #execute AIM 
                    aim_executions,last_timestamp=policy.aggressive_in_money(data,current_order,start_index,current_volume_profile_limits,current_priority)
                    self.executions.append(aim_executions)
                else:
                    #execute PIM
                    pim_executions,last_timestamp=policy.passive_in_money(data,current_order,start_index,current_volume_profile_limits,current_priority)
                    self.executions.append(pim_executions)

            


# # Creating Instance of Execution Strategy

# In[429]:


execution_strategy=ExecutionStrategy(data,urgency)


# # Executing Orders

# In[430]:


execution_strategy.execute_orders()


# # Final Output: Trade Log

# In[431]:


execution_strategy.executions


# # Market Making Implementation

# ### (See Part II Notebook )

'''
This is going to be a libary for a monte carlo simulation. Mostly going to be used in finance setting
-didn't like the current offerings out there
'''

import pandas as pd
import numpy as np
from concurrent import futures 

import matplotlib.pyplot as plt

class montecarlo:

    '''
    Monte Carlo class: 
    '''

    def __init__(self,returns_iter,  periods ,**kwargs):
    # starting_value = 0,sims=10000, action = "multiply" , cores = 1, workers = None , seed = None):
        
        '''
        Parameters

        -REQUIRED BEG-

        returns_iter : list, numpy array, pandas series
            some iterable/indexed array that works with numpy.random.choice(self.returns_iter)

        starting_value : int, float
            value that we will start with

        periods : int
            number of rows/runs the class will eventually produce

        -REQUIRED END-

        sims : int (optional)
            number of columns/simulations the class will eventually produce (default is 10000)

        action : str (optional)
            how are the numbers interacting (default is multiply), should only be multiply or add

        seed : int or 1-d array_like (optional)
            seed for numpy.random.seed

        '''

        self.returns_iter = returns_iter
        self.periods = periods
        
        self.starting_value = kwargs.get('starting_value',0)
        self.sims = kwargs.get('sims',10000)
        self.action = kwargs.get('action','multiply')
        self.max_workers = kwargs.get('max_workers',5)

        self.seed = kwargs.get('seed',None)
        
        if self.seed:
            np.random.seed = self.seed

        # self.simulatations = self.__run_monte()
    
    def run_monte(self):
        '''
        default run monte carlo, combines simulations as they are ran
        '''
        
        with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    
            completed = executor.map(self.__get_next_sim,range(self.sims))

        self.simulatations = pd.concat(completed, axis=1 ,copy=False)
    
        return self.simulatations

    def __get_next_sim(self,simulation_requests):

        '''
        makes a dataframe of returns pull randomly from returns_iter, then cum_sum() or sumprod() accordingly
        '''

        random_returns = [self.__get_random_factor() for period in range(self.periods)]
        random_returns.insert(0,self.starting_value)

        df_sim = pd.DataFrame(random_returns,columns = ['returns'])

        if self.action == 'add':
            df_sim[f'run_{simulation_requests}'] = df_sim.cumsum()
        elif self.action == 'multiply':
            df_sim[f'run_{simulation_requests}'] = df_sim.cumprod()

        return df_sim[f'run_{simulation_requests}']
        
    def get_simulations(self):

        return self.simulatations

    # def __run_monte_goal_stop(self):

    #     '''
    #     run monte carlo when goal_stop is False
    #     '''

    #     self.__start_df = {}        
        
    #     for simulation_requests in range(1,self.sims+1):
    #         self.__start_df[f'run_{simulation_requests}'] = list(self.starting_value)
        
    #     self.df = pd.DataFrame(data = self.starting_value) 

    #     for _ in self.periods:
    #         self.df = pd.concat([self.df, self.get_next_row], ignore_index=True)

    # def get_next_row(self):

    #     data_for_new_row = {}

    #     if self.action == 'add':

    #         for simulation_requests in range(1,self.sims+1):
    #             data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1] + self.__get_random_factor

    #             if goal_stop:

    #             if self.__check_min_max(data_for_new_row[f'run_{simulation_requests}']):
    #                 data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1]


    #     if self.action == 'multiply':

    #         for simulation_requests in range(1,self.sims+1):
    #             data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1] * self.__get_random_factor


    #             if self.__check_min_max(data_for_new_row[f'run_{simulation_requests}']): 
    #                 data_for_new_row[f'run_{simulation_requests}'] = self.df[f'run_{simulation_requests}'].iloc[-1]


    #     return pd.DataFrame(data = data_for_new_row)

    
    def __get_random_factor(self):
        '''
        returns a randomly selected 

        may implement other features: no replacement
        '''

        return np.random.choice(self.returns_iter)

    # def __check_min_max(self,data_for_new_row):
    #     '''
    #     check if new row has breach user specifeied min and or max
    #     '''

    #     if self.min_dd and data_for_new_row < self.min_dd:
    #         return False

    #     if self.max_dd and data_for_new_row > self.max_dd:
    #         return False

    def paramaters(self):

        paramaters_used = {'returns used' : self.returns_iter,  
                            'start' : self.starting_value, 
                            'number of simulations' : self.sims, 
                            'period per simulation' : self.periods ,
                            'function used (add or multiply)' : self.action, 
                            'seed':np.random.seed()
        }

        for key,item in paramaters_used.items():
            print(f'{key}: {item}')

        return paramaters_used

    def summary(self):

        summary_stats = {'max' : max(self.simulatations.max()),  
                         'min' : min(self.simulatations.min()), 
                         'avg end' : np.mean(self.simulatations.iloc[-1,:]),
                         'avg run' : self.__average_simulation()
                        }

        for key,value in summary_stats.items():
            print(f'{key} {value}')

        return summary_stats

    def __average_simulation(self):
        
        self.average_simulation = self.simulatations.apply(np.mean,axis=1)

        return self.average_simulation
        

    def plot(self,figsize = None):

        '''
        plots result of monte_carlo, basically gonna look like a rainbow
        '''

        if figsize:
            plt.figure(figsize = figsize)

        plt.title(f'{self.sims} Runs')

        plt.xlabel('Period')
        plt.ylabel('Value')

        with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor: 
            executor.map(self.__plot,self.simulatations.columns)

        plt.axhline(self.starting_value, color='black')
        plt.legend(self.simulatations.columns,loc = 'best')
        plt.show()

        return None
    
    def __plot(self,column):

        plt.plot(self.simulatations[column])

        return None

    # def dash_plot(self):

    #     app = dash.Dash(__name__)

    #     app.layout = html.Div(

    #         dcc.Graph(
    #             id='montecarlo_graph',
    #             figure={
    #                 'data': [{'x' : self.df.index(),
    #                           'y' : self.df[f'{column}'],
    #                           'text' : f'{column}',
    #                           'name' : f'{column}'}] for column in self.df

    #                     }

    #         )
    #         )

    #     return app
        #if __name__ == '__main__':
        #    app.run_server(debug=True)

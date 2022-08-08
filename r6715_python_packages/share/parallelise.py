# -*- coding: iso-8859-1 -*-
from __future__ import print_function
import functools
import multiprocessing as mp
import sys

'''
 Function to wrap the python map or multiprocessing map functions
 to make it easier to switch between the two. Also takes care of
 additional keyword arguments.
 - KW
'''

def wrap_map(func, func_arg, parallel=True, processes=None, **kwargs):
    '''
    Wraps the map function.
    Makes it easy to switch between parallel and non-parallel
    runs for efficiency, debugging and profiling.
    
    KWargs:
     * func: function to apply to an iterable
     * func_arg: iterable (second argument of the map function)
     * processes: number of processes used if parallel=True.
       If parallel=True and processes is None, then sets to the minimum
       of value out of (a) the length of func_arg or (b) number of cpus.
     * parallel: if True, uses pool.map. If False uses 
       the python built-in function map.
     * all other (**kwargs) get put into a partial function, to use in
       the map function.    
    '''
    
    partial_func = functools.partial(func, **kwargs)

    if parallel:
        if processes is None:
            processes = min(len(func_arg), mp.cpu_count())

        pool = mp.Pool(processes=processes)

        print('just about to do pool.map')    
        sys.stdout.flush()
        
        res = pool.map(partial_func, func_arg)

        print('just done pool.map')    
        sys.stdout.flush()
        
        pool.close() 
        pool.join()
    else:
        res = list(map(partial_func, func_arg))
        
    return res 

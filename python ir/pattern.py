# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:46:33 2019

@author: Soumo
"""
import threading
import time
def print_thread(ch,n):
    a=int((n+1)/2)
    for i in range(a):
        for k in range(a-i-1,0,-1):
            print(" ", end='\0')
        for j in range(2*i+1):
            print("{}".format(ch), end='\0')
        print("")
        
    for i in range(a-1):
        for k in range(i+1,0,-1):
            print(" ", end='\0')
        for j in range(2*(a-1-i)-1):
            print("{}".format(ch), end='\0')
        print("")

def call_thread(t=3,n=5):
    for i in range(t*5):
        print("\033[H\033[J")
        n=n+2
        t1 = threading.Thread(target=print_thread, args=('*',n))
        t1.start()
        time.sleep(0.1)
        t1.join()
        
    for i in range(t*5):
        print("\033[H\033[J")
        n=n-2
        t1 = threading.Thread(target=print_thread, args=('*',n))
        t1.start()
        time.sleep(0.1)
        t1.join()
        
if __name__ == "__main__": 
#    n=int(input("Enter diameter : "))
#    t=int(input("Enter time : "))
    t1 = threading.Thread(target=call_thread)
    t1.start()
#!/usr/bin/env python3

#   Usage:
#
#   1. Enter parameters below:
#       (1) Select mode: 1 or 2
#           mode 1: Every second has a probablilty of BOOM
#           mode 2: Use a pre-defined BOOM time range in seconds. (DEFAULT)
#       (2) Enter reward per second
#       (3) Enter refresh rate (By default, reward is refreshed every 0.1 second)
#
#   2. Press 'enter' to pause, and hand to the next player.


# ----------------------------------------------------------
# Parameters:
BOOM_MODE=2
REWARD_PER_SECOND=10
REFRESH_RATE=0.1

#BOOM MODE: 1 parameter
BOOM_PROB_SEC=0.1
#BOOM MODE: 2 parameter
BOOM_RANGE_LOWER=2
BOOM_RANGE_UPPER=5
# ----------------------------------------------------------




from time import localtime,sleep
from datetime import datetime
import numpy as np
import queue
import threading

boom_reward=np.random.uniform(BOOM_RANGE_LOWER*REWARD_PER_SECOND,
                                                     BOOM_RANGE_UPPER*REWARD_PER_SECOND)

def BOOM():
    if(BOOM_MODE == 1):
        if(np.random.rand() < BOOM_PROB_SEC ):
            return 1

    if(BOOM_MODE == 2):
        if(cumulative_reward+rwd>boom_reward):
            return 1
            

BOOM_PROB_SEC*=REFRESH_RATE

# Thread to detect 'enter'
def get_input(queue):
    detected=input()
    queue.put(detected)

queue_obj=queue.Queue()
Thread = threading.Thread(target=get_input, args=(queue_obj,), daemon=True)
Thread.start()

cumulative_reward=0
start_time=datetime.now().timestamp()

while(1):
    rwd=(datetime.now().timestamp()-start_time)*REWARD_PER_SECOND

    print("$",end='')
    print(round(rwd,2),end="\r")
    sleep(REFRESH_RATE)

    if(BOOM()):
        break

    if(queue_obj.qsize()):  
        input("Paused.")
        queue_obj.get()
        Thread = threading.Thread(target=get_input, args=(queue_obj,), daemon=True)
        Thread.start()
        cumulative_reward+=rwd
        start_time=datetime.now().timestamp()

print("\nBOOM!")


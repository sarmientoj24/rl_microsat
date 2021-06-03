import time


def get_remaining_time(max_iter, start_time, curr_iter):
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = (elapsed_time / (curr_iter + 1)) * (max_iter - (curr_iter + 1))
    
    print("Remaining time: ", remaining_time)
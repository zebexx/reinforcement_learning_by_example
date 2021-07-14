import numpy as np




test_array = np.genfromtxt("history/action_history.csv", delimiter=",")
print(len(test_array))


index = 0
for x in test_array:
    if x[0] == 0 and x[1] == 0:
        print(x)
        print(index)
        break
    index += 1
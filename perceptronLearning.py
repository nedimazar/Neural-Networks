import numpy as np
import pandas as pd

def getOutput(inputs, ws):
    out = sum([a * b for a, b in zip(inputs, ws)])
    return out > 0 

def perceptronLearning(data, alpha):
    # Initializing each weight to  a random number
    ws = [np.random.random() for x in range(data.shape[1]-1)]
    
    # Repeat to convergence
    while True:
        startingWs = ws.copy()
        # For each data point we will update the weights
        for x in range(data.shape[0]):
            # Getting the inputs and output from the row
            inputs = np.array(data.iloc[x,:-1])
            expected = np.array(data.iloc[x, -1])

            # Computing the output with the current weights
            actual = getOutput(inputs, ws)

            # Getting the error
            error = expected - actual

            # Updating the weights accordingly
            ws = [w + alpha * tin * error for w, tin in zip(ws, inputs)]
        
        # Convergence check
        if ws == startingWs:
            return ws


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    ws = perceptronLearning(data, alpha=0.1)

    predictions = []

    for x in range(data.shape[0]):
            inputs = np.array(data.iloc[x,:-1])
            a = np.array(inputs)
            predictions.append(1 * (np.sum(a*ws) > 0))


    print(f"\nWeights for perceptron learning: {np.array(ws)}\n")
    print(f"Training data true Y          : {np.array(data['out'])}")
    print(f"Predicted Y for trainging data: {np.array(predictions)}\n")



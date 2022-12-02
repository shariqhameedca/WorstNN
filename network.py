# It contains three functions 

# Predict an instance
def predict(network, input):
    output = input
    
    for layer in network:
        output = layer.forward(input)

    return output

# Train the model
def train(network, loss, loss_drv, X, y, epochs, lr):
    for epoch in epochs:
        cost = 0
        for x, y in zip(X, y):
            predicted = predict(network, x)
            
            cost += loss(y, predicted)
            
            grad = loss_drv(y, predicted)
            for layer in reversed(network):
                grad = layer.backward(grad, lr)

        cost /= len(X)

# Test the model on a test set
def test(network, loss, X_test, y_test):
    test_error = 0
    for x, y in zip(X_test, y_test):
        predicted = predict(network, x)
        test_error += loss(y, predicted)

    test_error /= len(X_test)
    return test_error


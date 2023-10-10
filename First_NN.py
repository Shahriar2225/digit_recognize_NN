from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import random
    


images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5 , 0.5 , (20,784))
w_h_o = np.random.uniform(-0.5, 0.5, (10,20))
b_i_h = np.zeros((20,1))
b_h_o = np.zeros((10,1))

learn_rate = 0.01
nr_correct =0
epochs = 3

for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)   # making vector to matirx
        l.shape += (1,)
        
        # forword propagation input -> hidden layer
        h_pre = w_i_h @ img + b_i_h
        h= 1/(1+np.exp(-h_pre))  # activation function in this case sigmoind
        
        # forword propagation hidden -> output
        
        o_pre = w_h_o @ h + b_h_o
        out = 1/(1+np.exp(-o_pre)) # activation function in this case sigmoind
        
        
        #cost/error function to find accuracy
        
        e = 1/len(out) * np.sum((out-l)**2 ,axis = 0)
        nr_correct += int(np.argmax(out) == np.argmax(l)) # it compare the 0th and 1th label
        
        #backpropagation output -> hidden (cost function derivative)
        delta_o = out - l #cost function derivative this is for MSE function only
        w_h_o += -learn_rate*delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        
        #back propagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o*(h*(1-h))
        w_i_h += -learn_rate*delta_h @ np.transpose(img)
        b_i_h += -learn_rate*delta_h
        
    # show accuracy for this epoch
    
    print(f"Accuracy : {round(nr_correct/images.shape[0]*100, 2)}%")
    nr_correct = 0

# Show result
t= True
while t:
    test = 0
    img = images[np.random.randint(0,59999)]
    
    plt.imshow(img.reshape(28,28), cmap = "Grays")

    img.shape += (1,)
    #forword propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784,1)
    h = 1/ (1+ np.exp(-h_pre))
    
    #forword propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    out = 1/(1+np.exp(-o_pre))
    
    plt.title (f"Prediction {out.argmax()}")
    plt.show()
    test+=1
    if(test <= 5):
        t=False
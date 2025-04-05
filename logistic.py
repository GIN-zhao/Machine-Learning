import numpy as np 

class LogisticRegression:
    def __init__(self,x,y,num_iterations=1000,learning_rate=1e-5):
        self.x = x 
        self.y = y 
        self.num_iterations = num_iterations 
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.x.shape[1])
        self.bias = 0.0 
        self.verbose = True 
        
    def sigmoid(self,x):
        z=  1.0/1.0+np.exp(-x)
        return z 
    def loss(self,x_samples_data,y_samples_data,y_samples_pred):
        loss_ = -1/len(x_samples_data) *(y_samples_data*np.log(y_samples_pred) +(1-y_samples_data)*np.log(1-y_samples_pred))
        
        return loss_  
    
    def gradient(self,x_samples_data,y_samples_data,y_samples_pred):
        dw = [0.0]*len(self.weights)
        db = 0.0 
        for i in range(len(x_samples_data)):
            dw = dw + (y_samples_data[i]-y_samples_pred[i])*x_samples_data[i]
            db = db + (y_samples_data[i]-y_samples_pred[i])
        dw = dw/len(x_samples_data)
        db = db/len(x_samples_data)
        return dw,db
    def predict(self,x_samples_data):
        y_samples_pred = self.sigmoid(np.dot(x_samples_data,self.weights)+self.bias)
        y_samples_pred = np.where(y_samples_pred>0.5,1,0)
        return y_samples_pred
    def train(self):
        for i in range(self.num_iterations):
            y_pred = self.predict(self.x)
            # print(y_pred)
            dw,db = self.gradient(self.x,self.y,y_pred)
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db
            if self.verbose:
                loss = self.loss(self.x,self.y,y_pred)
                print(f"Iteration {i} Loss: {loss}")
                
def main():
    x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
    y = np.array([0,1])
    model = LogisticRegression(x,y)
    model.train()
    print(model.predict(x))
main()
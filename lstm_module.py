import numpy as np

class LSTM_SS:
    
    def __init__(self, x_dim, y_dim, hidden_units, num_lstm_cell, learning_rate):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_units = hidden_units
        self.num_lstm_cell = num_lstm_cell
        self.learning_rate = learning_rate
        
        self.forget_list = [np.zeros((hidden_units,1)) for _ in range(num_lstm_cell)]
        self.hidden_list = [np.zeros((hidden_units,1)) for _ in range(num_lstm_cell)]
        self.candidate_list = [np.zeros((hidden_units,1)) for _ in range(num_lstm_cell)]
        self.cell_list = [np.zeros((hidden_units,1)) for _ in range(num_lstm_cell)]
        self.input_list = [np.zeros((hidden_units,1)) for _ in range(num_lstm_cell)]
        self.output_list = [np.zeros((hidden_units,1)) for _ in range(num_lstm_cell)]
        
        self.wf = np.random.random((hidden_units, x_dim+hidden_units)) #/np.sqrt((x_dim+self.hidden_units))
        self.wi = np.random.random((hidden_units, x_dim+hidden_units)) #/np.sqrt((x_dim+self.hidden_units))
        self.wc = np.random.random((hidden_units, x_dim+hidden_units)) #/np.sqrt((x_dim+self.hidden_units))
        self.wo = np.random.random((hidden_units, x_dim+hidden_units)) #/np.sqrt((x_dim+self.hidden_units))
        
        self.bf = np.random.random((hidden_units,1))
        self.bi = np.random.random((hidden_units,1))
        self.bc = np.random.random((hidden_units,1))
        self.bo = np.random.random((hidden_units,1))
        
        self.final_w = np.random.random((y_dim, hidden_units))
        self.final_b = np.random.random((y_dim,1))
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    def sigmoid_d(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def tanh(self, z):
        return np.tanh(z)
    
    def tanh_d(self, z):
        return 1 - np.square(np.tanh(z))
    
    
    def forward(self, x):
        x = np.array(x)
        self.x = x
        
        for t in range(1, self.num_lstm_cell):
            h_x_merged = np.vstack((self.hidden_list[t-1],self.x[t]))
            forget_gate = self.sigmoid(self.wf @ h_x_merged + self.bf)
            input_gate = self.sigmoid(self.wi @ h_x_merged + self.bi)
            output_gate = self.sigmoid(self.wo @ h_x_merged + self.bo)
            candidate_state = self.tanh(self.wc @h_x_merged + self.bc)
            cell_state = (forget_gate * self.cell_list[t-1]) + (input_gate * candidate_state)
            hidden_state = output_gate * self.tanh(cell_state)
            
            self.forget_list[t] = forget_gate
            self.hidden_list[t] = hidden_state
            self.cell_list[t] = cell_state
            self.candidate_list[t] = candidate_state
            self.output_list[t] = output_gate
            self.input_list[t] = input_gate
            
        y_pred = self.final_w @ self.hidden_list[-1] + self.final_b
        return y_pred
    
    
    def backpropagate(self, y_exp, y_pred):
        
        delta_hidden_list = [np.zeros((self.hidden_units,1)) for _ in range(self.num_lstm_cell+1)]
        delta_input_list = [np.zeros((self.hidden_units,1)) for _ in range(self.num_lstm_cell+1)]
        delta_output_list = [np.zeros((self.hidden_units,1)) for _ in range(self.num_lstm_cell+1)]
        delta_cell_list = [np.zeros((self.hidden_units,1)) for _ in range(self.num_lstm_cell+1)]
        delta_candidate_list = [np.zeros((self.hidden_units,1)) for _ in range(self.num_lstm_cell+1)]
        delta_forget_list = [np.zeros((self.hidden_units,1)) for _ in range(self.num_lstm_cell+1)]
        
        delta_final_w = np.zeros_like(self.final_w)
        delta_final_b =np.zeros_like(self.final_b)
        
        delta_wf = np.zeros_like(self.wf)
        delta_bf = np.zeros_like(self.bf)
        
        delta_wi = np.zeros_like(self.wi)
        delta_bi = np.zeros_like(self.bi)
        
        delta_wo = np.zeros_like(self.wo)
        delta_bo = np.zeros_like(self.bo)
        
        delta_wc = np.zeros_like(self.wc)
        delta_bc = np.zeros_like(self.bc)
        
        delta_e = y_exp - y_pred
        
        delta_final_w = delta_e * self.hidden_list[-1].T
        delta_final_b = delta_e
        
        for t in reversed(range(self.num_lstm_cell)):
            
            delta_hidden_list[t] = self.final_w.T @ delta_e + delta_hidden_list[t+1]
            delta_output_list[t] = self.tanh(self.cell_list[t]) * delta_hidden_list[t] * self.sigmoid_d(self.hidden_list[t])
            delta_cell_list[t] = self.output_list[t] * delta_hidden_list[t] * self.tanh_d(self.cell_list[t]) + delta_cell_list[t+1]
            delta_forget_list[t] = self.cell_list[t-1] * delta_cell_list[t] * self.sigmoid_d(self.forget_list[t])
            delta_input_list[t] = self.candidate_list[t] * delta_cell_list[t] * self.sigmoid_d(self.input_list[t])
            delta_candidate_list = self.input_list[t] * delta_cell_list[t] * self.tanh_d(self.candidate_list[t])
            
            z = np.vstack((self.hidden_list[t-1],self.x[t]))
            
            delta_wf += delta_forget_list[t] @ z.T
            delta_bf += delta_forget_list[t]
            
            delta_wi += delta_input_list[t] @ z.T
            delta_bi += delta_input_list[t]
            
            delta_wo += delta_output_list[t] @ z.T
            delta_bo += delta_output_list[t]
            
            delta_wc += delta_cell_list[t] @ z.T
            delta_bc += delta_cell_list[t]
            
        return delta_final_w, delta_final_b, delta_wf/self.num_lstm_cell, delta_bf/self.num_lstm_cell, delta_wi/self.num_lstm_cell, delta_bi/self.num_lstm_cell, delta_wo/self.num_lstm_cell, delta_bo/self.num_lstm_cell, delta_wc/self.num_lstm_cell, delta_bc/self.num_lstm_cell
    
    def fit(self, epochs, x, y_exp, verbose=True):
        
        training_loss_list = []
        
        for i in range(epochs):
            
            training_loss = 0
            
            for j in range(len(x)):
                y_pred = self.forward(x[j])
                
                delta_final_w, delta_final_b, delta_wf, delta_bf, delta_wi, delta_bi, delta_wo, delta_bo, delta_wc, delta_bc = self.backpropagate(y_exp[j], y_pred)
                
                self.final_w += self.learning_rate * delta_final_w
                self.final_b += self.learning_rate * delta_final_b
                
                self.wf += self.learning_rate * delta_wf
                self.bf += self.learning_rate * delta_bf
                
                self.wi += self.learning_rate * delta_wi
                self.bi += self.learning_rate * delta_bi
                
                self.wo += self.learning_rate * delta_wo
                self.bo += self.learning_rate * delta_bo
                
                self.wc += self.learning_rate * delta_wc
                self.bc += self.learning_rate * delta_bc
                
                training_loss += ((y_exp[j] - y_pred)**2)/2
                
            if verbose == True:
                print("Epochs: ", i+1 ,", Loss: ", training_loss[0][0]) 

            training_loss_list.append(training_loss)
            
        
    def predict(self, x):
            
        y_pred = []
            
        for i in range(len(x)):
            y_pred.append(self.forward(x[i]))
                
        return  np.concatenate(y_pred)
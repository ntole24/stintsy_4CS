import torch
import torch.nn as nn
import torch.nn.init

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, list_hidden, activation='relu'):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden
        self.activation = activation
        
        self.create_network()
        self.init_weights()

    def create_network(self):
        layers = []
        
        # Input layer to first hidden
        layers.append(nn.Linear(self.input_size, self.list_hidden[0]))
        layers.append(self.get_activation(mode=self.activation))

        # Hidden layers
        for i in range(len(self.list_hidden) - 1):
            layers.append(nn.Linear(self.list_hidden[i], self.list_hidden[i+1]))
            layers.append(self.get_activation(mode=self.activation))

        # Final Output Layer
        layers.append(nn.Linear(self.list_hidden[-1], self.num_classes))
        
        self.layers = nn.Sequential(*layers)

    def init_weights(self):
        gen = torch.manual_seed(2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1, generator=gen)
                nn.init.constant_(module.bias, 0)

    def get_activation(self, mode='relu'):
        if mode == 'tanh':
            return nn.Tanh()
        elif mode == 'relu':
            return nn.ReLU(inplace=True)
        return nn.Sigmoid()

    def forward(self, x, verbose=False):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if verbose:
                layer_type = layer.__class__.__name__
                print(f'Output of layer {i} ({layer_type}):')
                print(x, '\n')
        
        logits = x
        if self.num_classes == 1:
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=1)
            
        return logits, probabilities

    def predict(self, probabilities):
        if self.num_classes == 1:
            return (probabilities > 0.5).float()
        else:
            return torch.argmax(probabilities, dim=1)

    def fit(self, X_train_torch, y_train_torch, X_val_torch, y_val_torch, optimizer, criterion, max_epochs=400, convergence_threshold=0.00005, verbose=False):
        e = 0
        is_converged = False
        previous_loss = float('inf')
        train_losses = []
        val_losses = []

        y_train_all = y_train_torch.view(-1, 1).float()
        y_val_all = y_val_torch.view(-1, 1).float()

        while e < max_epochs and not is_converged:
            # Training Mode
            self.train()
            optimizer.zero_grad()
            
            logits, _ = self.forward(X_train_torch)
            loss = criterion(logits, y_train_all)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            train_losses.append(current_loss)

            # Validation Mode
            self.eval()
            with torch.no_grad():
                val_logits, val_probs = self.forward(X_val_torch)
                v_loss = criterion(val_logits, y_val_all).item()
                val_losses.append(v_loss)
                
                val_preds = self.predict(val_probs)
                val_acc = (val_preds == y_val_all).float().mean()

            # Logging
            if ((e + 1) % 10 == 0 or e == 0) and verbose:
                print(f'Epoch: {e+1} | Train Loss: {current_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {val_acc:.2%}')
            
            # Convergence Check
            if abs(previous_loss - current_loss) < convergence_threshold:
                print(f"Converged at epoch {e+1}")
                is_converged = True
            else:
                previous_loss = current_loss
                e += 1
                
        return train_losses, val_losses, e+1
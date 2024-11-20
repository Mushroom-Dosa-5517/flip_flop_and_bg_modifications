import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from jk_flip_flop import FF
# Constants for the task
CHARS = ['1', 'A', 'X', '2', 'B', 'Y', 'C', 'D']  # Possible characters
TARGETS = [('1', 'A', 'X'), ('2', 'B', 'Y')]  # Target sequences

# Dataset definition
class OneAX2BYDataset(Dataset):
    def __init__(self, num_sequences=1000, sequence_length=32):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.data = self.generate_sequences()

    def generate_sequences(self):
        sequences = []
        labels = []
        for _ in range(self.num_sequences):
            is_target = random.choice([True, False])
            if is_target:
                target_seq = random.choice(TARGETS)
                seq = list(target_seq) + [random.choice(CHARS) for _ in range(self.sequence_length - len(target_seq))]
                label = 1  # Target sequence label
            else:
                seq = [random.choice(CHARS) for _ in range(self.sequence_length)]
                while any(seq[i:i + 3] in TARGETS for i in range(len(seq) - 2)):
                    seq = [random.choice(CHARS) for _ in range(self.sequence_length)]
                label = 0  # Non-target sequence label
            sequences.append(seq)
            labels.append(label)
        char_to_idx = {ch: idx for idx, ch in enumerate(CHARS)}
        sequences = torch.tensor([[char_to_idx[ch] for ch in seq] for seq in sequences], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        return list(zip(sequences, labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Value Function and Q-Learning classes

class ValueFunction(nn.Module):
    def __init__(self, striatum_units):
        super(ValueFunction, self).__init__()
        # Make VF a learnable parameter
        self.VF = nn.Parameter(torch.zeros(striatum_units))  # Convert to learnable parameter

    def forward(self):
        return self.VF

class QValueRL(nn.Module):
    def __init__(self, action_space):
        super(QValueRL, self).__init__()
        # Make q_table a learnable parameter
        self.q_table = nn.Parameter(torch.zeros(action_space))  # Convert to learnable parameter
        self.gamma = 0.5

    def forward(self):
        return self.q_table
def select_action(q_values, epsilon=0.1):
    if random.random() < epsilon:
        # Exploration: random action
        action = random.randint(0, len(q_values) - 1)
    else:
        # Exploitation: action with highest Q-value
        action = torch.argmax(q_values).item()
    return action

# Custom LSTM cell for STN-GPe dynamics
class CustomLSTMCell(nn.Module):
    def __init__(self, units, epsilon_gpe_stn, distance_matrix):
        super(CustomLSTMCell, self).__init__()
        self.units = units
        self.lstm_cell = nn.LSTMCell(units, units)
        self.W_lateral = torch.exp(-distance_matrix / epsilon_gpe_stn).to(torch.float32)
        
    def forward(self, x, h, c):
        h_next, c_next = self.lstm_cell(x, (h, c))
        h_next = h_next.unsqueeze(-1)
        W_lateral_expanded = self.W_lateral.expand(h_next.size(0), -1, -1)
        h_next = torch.bmm(W_lateral_expanded, h_next).squeeze(-1)
        h_next = torch.tanh(h_next)
        return h_next, c_next

class ModifiedSTNGPe(nn.Module):
    def __init__(self, stn_gpe_units, epsilon_gpe_stn=0.1):
        super(ModifiedSTNGPe, self).__init__()
        distance_matrix = self.generate_distance_matrix(stn_gpe_units)
        self.stn_gpe_cell = CustomLSTMCell(stn_gpe_units, epsilon_gpe_stn, distance_matrix)

    def generate_distance_matrix(self, units):
        grid_size = int(units**0.5)
        distance_matrix = torch.zeros(units, units)
        for i in range(units):
            for j in range(units):
                xi, yi = divmod(i, grid_size)
                xj, yj = divmod(j, grid_size)
                distance_matrix[i, j] = (xi - xj)**2 + (yi - yj)**2
        return distance_matrix

    def forward(self, x, h_stn, c_stn):
        h_stn, c_stn = self.stn_gpe_cell(x, h_stn, c_stn)
        return h_stn, c_stn

# Main model class

class BasalGangliaRLModel(nn.Module):
    def __init__(self, input_dim, striatum_units, stn_gpe_units, gpi_units, action_space, epsilon_gpe_stn=0.1, tau_gpi=1.0):
        super(BasalGangliaRLModel, self).__init__()
        
        self.tau_gpi = tau_gpi  # Time constant for GPi
        self.gpi_decay = nn.Parameter(torch.tensor(1.0))  # GPi decay parameter
        
        # Embedding for cortical input
        self.embedding = nn.Embedding(input_dim, striatum_units)
        
        # Striatum and D1/D2 pathways using Flip-Flop Neurons
        self.d1_neuron = FF(striatum_units)
        self.d2_neuron = FF(striatum_units)

        # STN-GPe Loop with Custom Recurrent Cell (Dynamic System)
        self.modified_stn_gpe = ModifiedSTNGPe(stn_gpe_units, epsilon_gpe_stn)

        # GPi Output Layer
        self.gpi = nn.Linear(gpi_units, 1)  # Output for action selection

        # Q-values and VF modules as learnable parameters
        self.q_rl = QValueRL(action_space)
        self.VF = ValueFunction(striatum_units)

    def forward(self, x_t, prev_d1, prev_d2, h_stn, c_stn):
        # Embed the input for the current timestep
        embedded = self.embedding(x_t)

        # Process D1 and D2 neurons (Striatum) for the current timestep
        d1_output = self.d1_neuron(embedded, prev_d1)
        d2_output = self.d2_neuron(embedded, prev_d2)

        # STN-GPe dynamics for the current timestep
        h_stn, c_stn = self.modified_stn_gpe(d2_output, h_stn, c_stn)

        # GPi Dynamics (approximation of the differential equation)
        lambda_d1 = torch.sigmoid(d1_output)  # Gain parameter for D1 pathway
        lambda_d2 = torch.sigmoid(d2_output)  # Gain parameter for D2 pathway
        
        # Compute GPi input based on D1 and STN-GPe outputs, with dynamic gains
        gpi_input = -self.gpi_decay * h_stn + lambda_d1 * d1_output + lambda_d2 * h_stn
        gpi_output = torch.sigmoid(self.gpi(gpi_input))  # Final decision-making output

        return gpi_output, d1_output, d2_output, h_stn, c_stn
# Training function
def train_test_basal_ganglia_rl(model, train_dataloader, test_dataloader, epochs=50, epsilon=0.1):
    optimizer_nn = optim.Adam(model.parameters(), lr=0.001)
    optimizer_rl = optim.Adam([model.q_rl.q_table], lr=0.0005)
    criterion_nn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in tqdm(train_dataloader):   
            optimizer_nn.zero_grad()
            optimizer_rl.zero_grad()

            batch_size, seq_len = x.size()
            h_stn = torch.zeros(batch_size, model.modified_stn_gpe.stn_gpe_cell.units)
            c_stn = torch.zeros(batch_size, model.modified_stn_gpe.stn_gpe_cell.units)
            prev_d1 = torch.zeros(batch_size, model.d1_neuron.units)
            prev_d2 = torch.zeros(batch_size, model.d2_neuron.units)

            # Loop through each time step in the sequence
            for t in range(seq_len):
                x_t = x[:, t]
                gpi_output, prev_d1, prev_d2, h_stn, c_stn = model(x_t, prev_d1, prev_d2, h_stn, c_stn)

            # Calculate TD error as the loss for Q-value and Value Function updates
            q_values = model.q_rl()
            VF_values = model.VF()
            rewards = ((gpi_output > 0.5).squeeze(1) == y).float()  
            for i in range(batch_size):
                current_action = select_action(q_values, epsilon=epsilon_gpe_stn)
                next_action = torch.argmax(q_values).item()
                td_error = rewards[i] + model.q_rl.gamma * q_values[next_action] - q_values[current_action]
                rl_loss = td_error.pow(2).mean()
                rl_loss.backward(retain_graph=True)
            
            next_action = torch.argmax(q_values).item()
            nn_loss = criterion_nn(gpi_output.squeeze(1), y)
            nn_loss.backward()
            optimizer_nn.step()
            optimizer_rl.step()
            total_loss += nn_loss.item() 
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(test_dataloader):
                batch_size, seq_len = x.size()
                h_stn = torch.zeros(batch_size, model.modified_stn_gpe.stn_gpe_cell.units)
                c_stn = torch.zeros(batch_size, model.modified_stn_gpe.stn_gpe_cell.units)
                prev_d1 = torch.zeros(batch_size, model.d1_neuron.units)
                prev_d2 = torch.zeros(batch_size, model.d2_neuron.units)

                for t in range(seq_len):
                    x_t = x[:, t]
                    gpi_output, prev_d1, prev_d2, h_stn, c_stn = model(x_t, prev_d1, prev_d2, h_stn, c_stn)

                predictions = (gpi_output > 0.5).squeeze(1).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
    

# Example usage
input_dim = 8
striatum_units = 20
gpi_units = 20
stn_gpe_units = 20
action_space = 2
epsilon_gpe_stn = 0.05

model = BasalGangliaRLModel(input_dim, striatum_units, stn_gpe_units, gpi_units, action_space, epsilon_gpe_stn)
train_dataloader = DataLoader(OneAX2BYDataset(num_sequences=8000, sequence_length=25), batch_size=32, shuffle=True)
test_dataloader = DataLoader(OneAX2BYDataset(num_sequences=2000, sequence_length=25), batch_size=32, shuffle=False)

train_test_basal_ganglia_rl(model, train_dataloader, test_dataloader, epochs=50)
torch.save(model.state_dict(), 'bg.pt')
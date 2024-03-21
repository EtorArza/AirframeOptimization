import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from interfaces import problem
from tqdm import tqdm as tqdm
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from mpmath import *
import sys
sys.path.append('other_src')
from knn_kl_divergence import naive_estimator, scipy_estimator, skl_estimator, skl_efficient
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

mp.dps = 500
feasible_and_unfeasible_sample_size = 36000
seed = 2
prob = problem('toy', 'ignore')


def check_gpu():
    print("\n----------gpu-check------------")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if torch.cuda.current_device() != -1:
            print("Tensors and models are being placed on GPU.")
        else:
            print("No tensors or models are being placed on GPU.")
    else:
        print("GPU is not available. PyTorch is using CPU.")
    print("----------gpu-check------------\n")



class nn_classifier(nn.Module):
    def __init__(self, prob: problem, seed):
        super(nn_classifier, self).__init__()
        self.prob = prob
        self.model_path = f'cache/classifier_{prob.problem_name}_{seed}.pth'
        self.rs = np.random.RandomState(seed)
        self.fc1 = nn.Linear(prob.dim, prob.dim*2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(prob.dim*2, prob.dim*2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(prob.dim*2, prob.dim*2)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(prob.dim*2, prob.dim)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(prob.dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.training = False

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if self.training:
            x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        if self.training:
            x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        if self.training:
            x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        if self.training:
            x = self.dropout4(x)
        x = self.sigmoid(self.fc5(x))
        return x

    def create_dataset(self, feasible_samples, unfeasible_samples):
        x = np.concatenate((feasible_samples, unfeasible_samples), axis=0)
        y = torch.cat((torch.ones(len(feasible_samples), 1), torch.zeros(len(unfeasible_samples), 1)), 0)
        return x, y

    def train_model(self, x_train, y_train, epochs, batch_size):
        validation_split=0.2
        if os.path.isfile(self.model_path):
            raise FileExistsError("Aborting model train, as", self.model_path, " exists, which is the trained model. Delete this file to retrain.")
        check_gpu()
        self.train()
        self.training = True
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.00025)
        
        # Split data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_split, random_state=self.rs)
        train_dataset = TensorDataset(torch.Tensor(x_train), y_train)
        val_dataset = TensorDataset(torch.Tensor(x_val), y_val)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            # Training loop
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            

            # Validation loop
            val_loss = 0.0
            correct = 0
            total = 0
            self.training = False
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predicted = torch.round(outputs)  # Assuming threshold at 0.5
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            self.training = True
            validation_error = 1 - correct / total
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_dataloader):.3f}, Validation Loss: {val_loss/len(val_dataloader):.3f}, Validation Error: {validation_error*100.0:.3f}%")
        self.training = False
        print("Finished training.")
        torch.save

    def save_model_to_cache_folder(self):
        torch.save(self.state_dict(), self.model_path)


    def load_model_from_cache_folder(self):
        if os.path.isfile(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            self.eval()
            self.training = False
            print("Model loaded.")
        else:
            raise FileNotFoundError("No model found at specified path.")



class nn_encoding(nn.Module):
    def __init__(self, prob: problem, seed):
        super(nn_encoding, self).__init__()
        self.fc1 = nn.Linear(prob.dim, prob.dim)
        self.fc2 = nn.Linear(prob.dim, prob.dim)
        self.fc3 = nn.Linear(prob.dim, prob.dim)
        self.fc4 = nn.Linear(prob.dim, prob.dim)
        self.fc5 = nn.Linear(prob.dim, prob.dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._feasibility_function = prob.constraint_check
        self.n_constraints = prob.n_constraints
        self.problem_dim = prob.dim
        self.seed = seed
        self.rs = np.random.RandomState(seed)

        self.feasible_solutions = np.array([]).reshape((0,prob.dim))
        self.unfeasible_solutions = np.array([]).reshape((0,prob.dim))
        stacked_res_feasibility_check = np.array([]).reshape((0,self.n_constraints))
        stacked_random_sample = np.array([]).reshape((0,prob.dim))

        try:
            self.feasible_solutions = np.load(f'cache/feasible_{prob.problem_name}_{seed}.npy')
            self.unfeasible_solutions = np.load(f'cache/unfeasible_{prob.problem_name}_{seed}.npy')
            stacked_res_feasibility_check = np.load(f'cache/stacked_res_feasibility_check_{prob.problem_name}_{seed}.npy')
        except FileNotFoundError:
            # Generate feasible/unfeasible samples
            print("Loading feasible/unfeasible samples failed. Generating new samples:")
            pb = tqdm(total=feasible_and_unfeasible_sample_size)
            self.feasible_solutions = np.empty((feasible_and_unfeasible_sample_size, prob.dim))
            self.unfeasible_solutions = np.empty((feasible_and_unfeasible_sample_size, prob.dim))
            stacked_res_feasibility_check = np.empty((feasible_and_unfeasible_sample_size, self.n_constraints))
            stacked_random_sample = np.empty((feasible_and_unfeasible_sample_size, prob.dim))

            count_feasible = 0
            count_unfeasible = 0
            while min(count_feasible, count_unfeasible) < feasible_and_unfeasible_sample_size:
                random_sample = np.random.random((100, prob.dim))
                res_feasibility_check = self.feasibility_function(random_sample)
                new_feasible_solutions = random_sample[np.where(np.all(res_feasibility_check > 0.0, axis=1))]
                new_unfeasible_solutions = random_sample[np.where(np.all(res_feasibility_check < 0.0, axis=1))]
                for solution in new_feasible_solutions:
                    if count_feasible < feasible_and_unfeasible_sample_size:
                        self.feasible_solutions[count_feasible] = solution
                        count_feasible += 1
                for solution in new_unfeasible_solutions:
                    if count_unfeasible < feasible_and_unfeasible_sample_size:
                        self.unfeasible_solutions[count_unfeasible] = solution
                        count_unfeasible += 1
                pb.n = min(count_feasible, count_unfeasible)
                pb.refresh()
            pb.close()
            self.feasible_solutions = self.feasible_solutions[:feasible_and_unfeasible_sample_size,:]
            self.unfeasible_solutions = self.unfeasible_solutions[:feasible_and_unfeasible_sample_size,:]
            np.save(f'cache/feasible_{prob.problem_name}_{seed}.npy', self.feasible_solutions)
            np.save(f'cache/unfeasible_{prob.problem_name}_{seed}.npy', self.unfeasible_solutions)
            np.save(f'cache/stacked_res_feasibility_check_{prob.problem_name}_{seed}.npy', stacked_res_feasibility_check)



        self.std = np.std(stacked_res_feasibility_check, axis=0)
        self.mean = np.mean(stacked_res_feasibility_check, axis=0)
        self.q99_abs_unfeasible = np.quantile(np.abs(np.clip(stacked_res_feasibility_check, a_min=None, a_max=1e-5)), q=0.99, axis=0)


    def feasibility_function(self, x_matrix):
        return np.array([self._feasibility_function(el) for el in x_matrix])

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x
    
    def loss_constraints(self, output):
        feasibility_values = self.feasibility_function(output)
                
        # clip to 0 as max, because as constraints, the value does not matter as long as it is positive.
        res = np.clip(feasibility_values, a_min=None, a_max=1e-5) 

        # normalize such that sum is usually lower than 1
        res = np.sum(res / self.q99_abs_unfeasible) / feasibility_values.shape[0] / feasibility_values.shape[1]

        res = -res # loss function is minimized.
        return  res

    def loss_coverage(self, output):
        total_d = 0
        for v in output:
            distances = np.linalg.norm(self.feasible_solutions - v, axis=1)
            min_distance = np.min(distances)
            total_d += min_distance
        return total_d / output.shape[0]


    def validation_loss(self, input):
        output = self(input).detach().numpy()
        l_constraints = self.loss_constraints(output)
        l_coverage = self.loss_coverage(output)
        l_kl_div = self.loss_kl_div(output)

        print("loss constraints:", l_constraints, "|", "loss coverage:", l_coverage, "|", "loss kl_div:", l_kl_div)

        return (l_constraints, l_coverage, l_kl_div)

    def loss_kl_div(self, input):
        # wangDivergenceEstimationMultidimensional2009
        output = self(input).detach().numpy()
        return scipy_estimator(self.feasible_solutions, output)

    def set_parameters(self, params):
        idx = 0
        for param in self.parameters():
            param_length = np.prod(param.size())
            param.data.copy_(torch.tensor(params[idx:idx+param_length].reshape(param.size())))
            idx += param_length

    def get_parameters(self):
        params = []
        for param in self.parameters():
            params.extend(param.cpu().detach().numpy().flatten())
        return np.array(params)



    def train_constraints_network(self, epochs):
        pass




    def optimize_network_parameters(self, max_evaluations):
        pass




        exit(0)

        import torch
        from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

        model = Unet1D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 32
            # dim = problem_dim//2,
            # dim_mults = (1,),
            # resnet_block_groups=1,
            # channels = 1
        )

        diffusion = GaussianDiffusion1D(
            model,
            seq_length = problem_dim,
            timesteps = 1000,
            objective = 'pred_v'
        )

        training_seq = torch.rand(64, 1, problem_dim) # features are normalized from 0 to 1
        dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below


        # Define optimizer
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)  # Adjust learning rate as needed

        # Define training parameters
        epochs = 10  # Adjust number of epochs as needed

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            loss = diffusion(training_seq)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print training progress
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        exit(0)


        loss = diffusion(training_seq)
        loss.backward()



        # Or using trainer

        trainer = Trainer1D(
            diffusion,
            dataset = dataset,
            train_batch_size = 32,
            train_lr = 1.6e-4,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
        )
        trainer.train()

        # after a lot of training

        sampled_seq = diffusion.sample(batch_size = 1)
        sampled_seq.shape # (4, 32, 128)

        print(sampled_seq, sampled_seq.shape)

        exit(0)






        # directly

        input_data = torch.randn(batch_size, problem_dim)
        optimizer = optim.Adam(self.parameters())  # Using Adam optimizer
        criterion = nn.MSELoss()  # You can change the loss function as needed

        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero the gradients

            output = self(input_data)
            loss = criterion(output, torch.zeros_like(output))  # Example loss, change as needed
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

        print('Optimization finished.')




        # import nevergrad as ng
        # self.rs = np.random.RandomState(seed+78)
        # x0 = self.get_parameters()
        # print(x0.shape, type(x0))

        # param = ng.p.Instrumentation(ng.p.Array(lower=-5.0, upper=5.0, init=x0))
        # self.optimizer = ng.optimizers.NGOpt(parametrization=param, budget=max_evaluations, num_workers=1)

        # for i in range(max_evaluations):

        #     self.prev_sol = self.optimizer.ask()
        #     params = self.prev_sol[0][0].value
        #     print(param.shape)
        #     print(params.shape)
        #     self.set_parameters(params)
        #     input_data = torch.randn(batch_size, problem_dim)
        #     loss = model.kl_div_loss(input_data)
        #     self.optimizer.tell(self.prev_sol, loss)
        #     print(i, loss)




        exit(0)

        loss = model.kl_div_loss(input_data)



seed = 2
encoding_model = nn_encoding(prob, seed)
classifier_model = nn_classifier(prob, seed)

try:
    classifier_model.load_model_from_cache_folder()
    print("loaded model", classifier_model.model_path)

except FileNotFoundError:
    print("Training classifier from scratch.")
    x_train, y_train = classifier_model.create_dataset(encoding_model.feasible_solutions, encoding_model.unfeasible_solutions)
    classifier_model.train_model(x_train, y_train, epochs=100, batch_size=32)
    classifier_model.save_model_to_cache_folder()


for i in range(100):
    input = encoding_model.unfeasible_solutions[i]
    print("evaluation:",prob.constraint_check(input),  classifier_model(torch.tensor(input, dtype=torch.float)))

print("---")

for i in range(100):
    input = encoding_model.feasible_solutions[i]
    print("evaluation:",prob.constraint_check(input),  classifier_model(torch.tensor(input, dtype=torch.float)))


exit(0)
model.optimize_network_parameters(1000)








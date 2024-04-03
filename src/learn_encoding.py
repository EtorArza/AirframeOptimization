import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
import torch.nn.functional

feasible_and_unfeasible_sample_size = int(1e6)


classifier_epochs = 100
classifier_batch_size = 3200
classfier_learning_rate = 0.00025
classifier_learning_log = "classifier_training.log"


encoder_epochs = int(5e5)
encoder_leraning_rate = 0.00025
encoder_batch_size = 6400
batch_size_distances_mantained = 120
feasible_refs_sample_size_for_loss = 120
encoder_learning_log = "encoder_training.log"


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
    def __init__(self, prob, seed):
        super(nn_classifier, self).__init__()
        assert prob.dim >= 8, "the NN architecture in the classifier only makes sense for dim>8. For smaller dimensions, layer 7 should probably be removed."
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
        self.fc5 = nn.Linear(prob.dim, prob.dim // 2)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(prob.dim // 2, prob.dim // 4)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(prob.dim // 4, 1)
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
        x = torch.relu(self.fc5(x))
        if self.training:
            x = self.dropout5(x)
        x = torch.relu(self.fc6(x))
        if self.training:
            x = self.dropout6(x)
        x = self.sigmoid(self.fc7(x))
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
        optimizer = optim.Adam(self.parameters(), lr=classfier_learning_rate)
        
        # Split data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_split, random_state=self.rs)
        train_dataset = TensorDataset(torch.Tensor(x_train), y_train)
        val_dataset = TensorDataset(torch.Tensor(x_val), y_val)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_loss = 1e10

        for epoch in tqdm(range(epochs)):
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
                    val_loss += loss.item()/len(val_dataloader)
                    predicted = torch.round(outputs)  # Assuming threshold at 0.5
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model_to_cache_folder()
            self.training = True
            validation_error = 1 - correct / total
            with open(classifier_learning_log, "a") as file:
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_dataloader):.3f}, Validation Loss: {val_loss:.3f}, Validation Error: {validation_error*100.0:.3f}%, Best loss: {best_loss}", file=file)
        self.training = False
        print("Finished training.")


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
    def __init__(self, prob, seed):
        super(nn_encoding, self).__init__()
        self.fc1 = nn.Linear(prob.dim, prob.dim*2)
        self.fc2 = nn.Linear(prob.dim*2, prob.dim*2)
        self.fc3 = nn.Linear(prob.dim*2, prob.dim*2)
        self.fc4 = nn.Linear(prob.dim*2, prob.dim*2)
        self.fc5 = nn.Linear(prob.dim*2, prob.dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._feasibility_function = prob._constraint_check
        self.n_constraints = prob.n_constraints
        self.problem_dim = prob.dim
        self.seed = seed
        self.rs = np.random.RandomState(seed)
        self.model_path = f'cache/encoder_{prob.problem_name}_{seed}.pth'


        self.feasible_solutions = np.array([]).reshape((0,prob.dim))
        self.unfeasible_solutions = np.array([]).reshape((0,prob.dim))

        try:
            self.feasible_solutions = np.load(f'cache/feasible_{prob.problem_name}_{seed}.npy')
            self.unfeasible_solutions = np.load(f'cache/unfeasible_{prob.problem_name}_{seed}.npy')
        except FileNotFoundError:
            # Generate feasible/unfeasible samples
            print("Loading feasible/unfeasible samples failed. Generating new samples:")
            print(feasible_and_unfeasible_sample_size)
            pb = tqdm(total=feasible_and_unfeasible_sample_size)
            self.feasible_solutions = np.empty((feasible_and_unfeasible_sample_size, prob.dim))
            self.unfeasible_solutions = np.empty((feasible_and_unfeasible_sample_size, prob.dim))

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
        return np.mean(np.all(feasibility_values<0, axis=1)) # Return the proportion of unfeasible values.

    def get_coverage_batch_size(self):

        sample_size = 500
        loss = 1.0
        x = []
        y = []
        while sample_size*2 < self.feasible_solutions.shape[0]:
            sample_size += 500
            indices = self.rs.choice(self.feasible_solutions.shape[0], size=sample_size*2, replace=False)
            feasible_refs1 = torch.tensor(self.feasible_solutions[indices[:sample_size],:], dtype=torch.float32)
            feasible_refs2 = torch.tensor(self.feasible_solutions[indices[sample_size:],:], dtype=torch.float32)
            loss = self._loss_coverage_formula(feasible_refs1, feasible_refs2)
            x.append(sample_size)
            y.append(float(loss.cpu()))
        from matplotlib import pyplot as plt
        plt.plot(x, y)
        plt.show()


    def _loss_coverage_formula(self, output, references):
        return torch.mean(torch.min(torch.cdist(output, references), dim=0)[0])

    def loss_coverage(self, output):
        '''
        Average (over all outputs) of the minimum distance between the output and any of the feasible solutions. 
        '''
        indices = self.rs.choice(self.feasible_solutions.shape[0], size=feasible_refs_sample_size_for_loss, replace=False)
        feasible_refs = torch.tensor(self.feasible_solutions[indices,:], dtype=torch.float32)
        res = self._loss_coverage_formula(output, feasible_refs)
        return res
        
    def loss_distances_mantained(self, input, output):
        '''
        Minimize difference in distance between input and output.
        Compute difference in pairwise distance matrices, and compute the frobenius norm of the result.
        Minimizing 'distances_mantained_loss' implies the distances between the vectors in input and output are similar to each other.
        '''
        distances_mantained_loss = 0
        for idx_start in range(batch_size_distances_mantained, input.shape[0], batch_size_distances_mantained):
            idx_end = idx_start + batch_size_distances_mantained
            distances_mantained_loss += torch.mean(torch.abs(torch.cdist(input[idx_start:idx_end], input[idx_start:idx_end]) - torch.cdist(output[idx_start:idx_end], output[idx_start:idx_end])))
        distances_mantained_loss *= (batch_size_distances_mantained / input.shape[0])
        return distances_mantained_loss


    def validation_loss(self, input):
        output = self(input).detach().numpy()
        l_constraints = self.loss_constraints(output)
        l_coverage = self.loss_coverage(output)
        l_kl_div = self.loss_kl_div(output)

        print("loss constraints:", l_constraints, "|", "loss coverage:", l_coverage, "|", "loss kl_div:", l_kl_div)

        return (l_constraints, l_coverage, l_kl_div)

    def loss_kl_div(self, output):
        # wangDivergenceEstimationMultidimensional2009
        sample_size = 3600
        return skl_efficient(self.feasible_solutions[:sample_size,:], output[:sample_size,:])



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

    def validation_loss(self, epoch, classifier_model):
        n_reps = 50
        validation_batch_size = encoder_batch_size
        classifier_loss = 0
        coverage_loss = 0
        distances_mantained_loss = 0
        baseline_coverage_loss = 0
        with torch.no_grad():
            for i in range(n_reps):
                inputs = torch.rand((validation_batch_size, self.problem_dim))
                encoding_outputs = self(inputs)
                classifier_outputs = classifier_model(encoding_outputs)
                classifier_loss += torch.mean(classifier_outputs) / n_reps
                coverage_loss += self.loss_coverage(encoding_outputs) / n_reps

                indices = self.rs.choice(self.feasible_solutions.shape[0], size=validation_batch_size, replace=False)
                feasible_refs1 = torch.tensor(self.feasible_solutions[indices,:], dtype=torch.float32)
                baseline_coverage_loss += self.loss_coverage(feasible_refs1) / n_reps


                distances_mantained_loss += self.loss_distances_mantained(inputs, encoding_outputs) / n_reps
                total_loss = classifier_loss+(coverage_loss-baseline_coverage_loss)
            with open(encoder_learning_log, "a") as file:
                print(f'Epoch: {epoch} Validation Loss: {classifier_loss} + {coverage_loss - baseline_coverage_loss} = {total_loss}', file=file)
        return total_loss

    def train_model(self, classifier_model, epochs, batch_size):
        if os.path.isfile(self.model_path):
            raise FileExistsError("Aborting model train, as", self.model_path, " exists, which is the trained model. Delete this file to retrain.")
        check_gpu()
        self.train()
        best_loss = 1e10
        optimizer_encoding = optim.Adam(self.parameters(), lr=0.00025)
        for epoch in tqdm(range(epochs)):
            inputs = torch.rand((batch_size, self.problem_dim))
            encoding_outputs = self(inputs)
            classifier_outputs = classifier_model(encoding_outputs)
            loss = torch.mean(classifier_outputs) + self.loss_coverage(encoding_outputs) # + self.loss_distances_mantained(inputs, encoding_outputs)
            optimizer_encoding.zero_grad()
            loss.backward()
            optimizer_encoding.step()

            if epoch % 1000 == 0:
                val_loss = self.validation_loss(epoch, classifier_model)
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model_to_cache_folder()


    def save_model_to_cache_folder(self):
        torch.save(self.state_dict(), self.model_path)

    def load_model_from_cache_folder(self):
        if os.path.isfile(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            self.eval()
            print("Model loaded.")
        else:
            raise FileNotFoundError("No model found at specified path.")


# y = []
# x = []
# for size in tqdm(range(5,600, 20)):
#     num_matrices = 1000
#     random_matrices = torch.rand(num_matrices, size, size) - torch.rand(num_matrices, size, size)
#     frobenius_norms = torch.norm(random_matrices, p='fro', dim=(1, 2))
#     x.append(size)
#     y.append(float(torch.mean(frobenius_norms).cpu()))
# from matplotlib import pyplot as plt
# plt.plot(x, y)
# plt.show()
# exit(0)

class solution_space_encoder:

    def __init__(self, prob, seed):

        self.classifier_model = nn_classifier(prob, seed)
        self.encoding_model = nn_encoding(prob, seed)

        try:
            self.classifier_model.load_model_from_cache_folder()
            print("loaded model", self.classifier_model.model_path)

        except FileNotFoundError:
            with open(classifier_learning_log, "a") as file:
                print("-----------------------------", file=file)
                print(f"Training classifier from scratch. {prob.problem_name} {seed}", file=file)
            x_train, y_train = self.classifier_model.create_dataset(self.encoding_model.feasible_solutions, self.encoding_model.unfeasible_solutions)
            self.classifier_model.train_model(x_train, y_train, classifier_epochs, classifier_batch_size)
            self.classifier_model.load_model_from_cache_folder() # Load model with best validation error


        try:
            self.encoding_model.load_model_from_cache_folder()
            print("loaded model", self.encoding_model.model_path)
        except FileNotFoundError:
            with open(encoder_learning_log, "a") as file:
                print("-----------------------------", file=file)
                print(f"Training encoder from scratch. {prob.problem_name} {seed}", file=file)
            print("Training encoding from scratch.")
            self.encoding_model.train_model(self.encoding_model, encoder_epochs, encoder_batch_size)
            self.encoding_model.load_model_from_cache_folder() # Load model with best validation error


    def encode(self, input):
        assert input.shape == (self.classifier_model.prob.dim,)
        with torch.no_grad():
            res = self.encoding_model(torch.tensor(input, dtype=torch.float32)).cpu().numpy()
        return res
        
# import interfaces 
# prob = interfaces.problem('windflo', 100, 'ignore', 2)
# solution_space_encoder(prob, 2)






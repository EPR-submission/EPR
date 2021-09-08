import numpy as np
import torch
import torch.nn as nn

MAX_EXP = 9.0

def log(x):
    return torch.log(x.clamp(np.exp(-MAX_EXP)))

def sigmoid(x):
    return torch.sigmoid(x)

class ScoreModel(nn.Module):
    def __init__(self, num_labels=3, used_scores=["entailment", "contradiction", "neutral", "sim_mul", "sim_diff"], init_layer=True, input_dim=512*3):
        super().__init__()
        self.num_labels = num_labels
        self.used_scores = used_scores
        self.split_size = int(input_dim/3)

        linear = nn.Linear(len(used_scores), num_labels)
        if init_layer:
            weights = np.random.uniform(-0.01, 0.01,
                                        size=(num_labels, len(used_scores)))
            for i, score_name in enumerate(["entailment", "contradiction", "neutral"]):
                if score_name in self.used_scores:
                    weights[i, self.used_scores.index(score_name)] += 1.

            linear.weight = nn.Parameter(torch.from_numpy(weights).float())
            linear.bias = nn.Parameter(
                torch.from_numpy(np.random.uniform(-0.01, 0.01, size=num_labels)).float())
            print("Initial classifier weights:", linear.weight.data)
        self.classifier = linear
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
    
    def mean_var_normalize(self, X):
        m = X.mean(dim=-1).unsqueeze(-1).expand_as(X)
        s = X.std(dim=-1).unsqueeze(-1).expand_as(X)
        X = (X - m) / s * 6
        return X

    def _get_true_false(self, X):
        #half_size = int(X.size(1) / 2)
        X_true, X_false = torch.split(X, self.split_size, dim=-1)
        # X_true, X_false = X[:, :, ::2], X[:, :, 1::2]
        return X_true, X_false

    def compute_features(self, X, Y, conf_a, conf_b):
        features = []
        if "entailment" in self.used_scores:

            dim_probability = 1 - sigmoid(-Y) * sigmoid(X)
            dim_probability = log(dim_probability)
            dim_probability = dim_probability.mean(dim=-1)
            ent_score = torch.exp(dim_probability)
            ent_score = ent_score.unsqueeze(-1)
            features.append(ent_score)

        if "contradiction" in self.used_scores:

            X_true, X_false = self._get_true_false(X)
            Y_true, Y_false = self._get_true_false(Y)

            x = torch.stack((X_true, X_false), dim=-1)
            y = torch.stack((Y_true, Y_false), dim=-1)
            x_softmax = self.softmax(x)
            y_softmax = self.softmax(y)
            if conf_a == None or conf_b == None:
                x_c = 1
                y_c = 1
            else:
                x_c = sigmoid(conf_a)
                y_c = sigmoid(conf_b)
            X_true = x_c * x_softmax[:,:,0]
            X_false = x_c * x_softmax[:,:,1]
            Y_true = y_c * y_softmax[:,:,0]
            Y_false = y_c * y_softmax[:,:,1]
            S_k = X_true * Y_false + X_false * Y_true
        
            # S_k = sigmoid(X_true) * sigmoid(Y_false) +\
            #     sigmoid(X_false) * sigmoid(Y_true) -\
            #     sigmoid(X_true) * sigmoid(Y_false) * sigmoid(X_false) * sigmoid(Y_true)

            cont_prob_dim = (1 - S_k)
            cont_log_prob = log(cont_prob_dim)
            mean_cont_log_probs = torch.mean(cont_log_prob, dim=-1)

            cont_score = 1 - torch.exp(mean_cont_log_probs)
            cont_score = cont_score.unsqueeze(-1)

            features.append(cont_score)

        if "neutral" in self.used_scores:
            neutral_score = self.relu(1 - ent_score - cont_score)
            features.append(neutral_score)

        if "sim_mul" in self.used_scores:
            sim_mul_score = (sigmoid(X) * sigmoid(Y)).mean(-1)
            features.append(sim_mul_score.unsqueeze(-1))

        if "sim_diff" in self.used_scores:
            sim_diff_score = torch.abs(sigmoid(X) - sigmoid(Y)).mean(-1)
            features.append(sim_diff_score.unsqueeze(-1))
        
        # not sure it is helpful
        if "cat" in self.used_scores:
            cat_score = torch.cat([sigmoid(X), sigmoid(Y)], dim=-1).mean(-1)
            features.append(cat_score.unsqueeze(-1))

        return features
    
    def split_feature_confidence(self, X):
        t1, t2, t3 = torch.split(X, self.split_size, dim=-1)
        # t1, t2, t3 = X[:, :, ::3], X[:, :, 1::3], X[:, :, 2::3]
        return torch.cat([t1, t2], dim=-1), t3

    def forward(self, rep_a, rep_b):
        rep_a, conf_a = self.split_feature_confidence(rep_a)
        rep_b, conf_b = self.split_feature_confidence(rep_b)

        rep_a, rep_b = self.mean_var_normalize(
            rep_a), self.mean_var_normalize(rep_b)

        vectors_concat = self.compute_features(rep_b, rep_a, conf_a, conf_b)
        features = torch.cat(vectors_concat, -1)

        output = self.classifier(features)
        return output

class EmbeddingModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=512*3):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        x = self.linear(x)
        return x
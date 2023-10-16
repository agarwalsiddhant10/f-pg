import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from policy import MlpNetwork

class DiscreteActor(nn.Module):
    def __init__(self, input_dim, action_space, max_state=10.0, min_state=10.0):
        super().__init__()
        self.diff_state = np.array(max_state - min_state).astype(np.float32)
        self.mean_state = np.asarray(self.diff_state / 2 + min_state).astype(np.float32)
        self.input_dim = input_dim
        self.action_space = action_space

        self.model = MlpNetwork(input_dim=input_dim, output_dim=action_space)

    def normalize(self, x):
        # normalize the state to 0-1
        x = x.type(torch.float32)
        x = (x - self.mean_state) / self.diff_state
        return x

    def forward(self, states, deterministic=False):
        # return action and action log prob

        # normalize the states
        normalized_states = self.normalize(states)
        model_outputs = self.model(normalized_states)

        probs = nn.functional.softmax(model_outputs, dim=1)
        log_probs = nn.functional.log_softmax(model_outputs, dim=1)

        if deterministic:
            # use the maximum probabilty
            acts = torch.argmax(probs, 1)
            log_probs_sample = log_probs[torch.arange(states.shape[0]), acts]
            return acts, log_probs_sample

        else:
            # sample
            acts = probs.multinomial(1)
            log_probs_sample = log_probs[torch.arange(states.shape[0]), acts]
            return acts, log_probs_sample


    def log_probs(self, states, actions):
        # return the log probs of the actions

        # normalize the states
        normalized_states = self.normalize(states)
        model_outputs = self.model(normalized_states)

        log_probs = nn.functional.log_softmax(model_outputs, dim=1)
        # print(torch.arange(states.shape[0]))
        # print(actions.int().squeeze(-1))
        log_probs_sample = log_probs[torch.arange(states.shape[0]), actions.long().squeeze(-1)]

        return log_probs_sample


    def loss(self, states, actions, returns):
        # compute the loss function

        log_probs = self.log_probs(states, actions)
        loss =  -torch.mean(torch.mul(log_probs, returns))
        return loss


    def sample_action(self, states):
        # forward in eval mode
        with torch.no_grad():
            acts, log_probs = self.forward(states)
        return acts, log_probs

    def entropy(self, states):
        normalized_states = self.normalize(states)
        model_outputs = self.model(normalized_states)

        entropy = Categorical(logits = model_outputs).entropy()
        return entropy
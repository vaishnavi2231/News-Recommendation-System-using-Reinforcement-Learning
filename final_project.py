# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer

def load_news_data(news_path):
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        header=None,
        names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    )
    news_df = news_df[["news_id", "title"]]
    news_df.dropna(inplace=True)
    return news_df.set_index("news_id").to_dict()["title"]

def load_user_behaviors(behaviors_path):
    user_histories = defaultdict(list)
    user_impressions = []

    with open(behaviors_path, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            user_id = parts[1]
            history = parts[3].split() if parts[3] else []
            impressions = parts[4].split()
            impression_data = []
            for imp in impressions:
                nid, label = imp.split('-')
                impression_data.append((nid, int(label)))

            user_histories[user_id].extend(history)
            user_impressions.append((user_id, history, impression_data))

    return user_histories, user_impressions

news_path = "./data/news.tsv"
behaviors_path = "./data/behaviors.tsv"

news_titles = load_news_data(news_path)
user_histories, user_impressions = load_user_behaviors(behaviors_path)

model = SentenceTransformer("all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

'''We generate the embeddings once and load it the consecutive times'''
# news_embeddings = {nid: model.encode(title) for nid, title in news_titles.items()} 

with open('./news_embeddings.pkl', 'rb') as f:
    news_embeddings = pickle.load(f)

def get_user_embedding(user_click_history, news_embeddings, embedding_dim=384):
    vectors = [news_embeddings[nid] for nid in user_click_history if nid in news_embeddings]
    if not vectors:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)

class NewsRecommendationEnv(gym.Env):
    def __init__(self, news_embeddings, user_histories, user_impressions, num_articles=5):
        super().__init__()
        self.news_embeddings = news_embeddings
        self.user_histories = user_histories
        self.user_impressions = user_impressions
        self.embedding_dim = len(next(iter(news_embeddings.values())))
        self.num_articles = num_articles
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.embedding_dim * (1 + num_articles),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(num_articles)

    def reset(self):
        while True:
            self.current_user_id, self.user_click_history, self.current_impression = random.choice(self.user_impressions)
            if len(self.current_impression) >= self.num_articles:
                break
        sampled = random.sample(self.current_impression, self.num_articles)
        self.candidate_ids = [nid for nid, _ in sampled]
        self.labels = [label for _, label in sampled]
        self.user_embedding = get_user_embedding(self.user_click_history, self.news_embeddings, self.embedding_dim)
        self.article_embeddings = [self.news_embeddings[nid] for nid in self.candidate_ids]
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.user_embedding] + self.article_embeddings)

    def step(self, action):
        reward = self.labels[action]
        done = True
        return self._get_state(), reward, done, {}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        common_output = self.common(state)
        logits = self.actor(common_output)
        value = self.critic(common_output)
        return logits, value

class A2CAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, entropy_weight=0.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActorCritic(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_weight = entropy_weight

    def select_action(self, state, temperature=1.0):
        device = next(self.model.parameters()).device
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, _ = self.model(state)
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

class DQNAgent:
    def __init__(self, model_path, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.model(state_tensor).argmax().item()

st.title("\U0001F4F0 Real-Time News Recommendation (Deep Reinforcement Learning Simulation)")

agent_type = st.selectbox("Choose an Agent:", ["DQN", "A2C"])

if "env" not in st.session_state or st.session_state.get("last_agent") != agent_type:
    st.session_state.env = NewsRecommendationEnv(news_embeddings, user_histories, user_impressions, num_articles=5)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.candidate_ids = st.session_state.env.candidate_ids
    st.session_state.last_agent = agent_type

env = st.session_state.env
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

if agent_type == "DQN":
    agent = DQNAgent("dqn_model.pth", state_size, action_size)
else:
    agent = A2CAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load("a2c_agent.pth", map_location=torch.device('cpu')))
    agent.model.eval()

st.subheader("Candidate News Articles:")
for idx, nid in enumerate(st.session_state.candidate_ids):
    st.write(f"Article {idx}: {news_titles.get(nid, 'Unknown Title')}")

if st.button("\U0001F4C8 Recommend Best Article"):
    state = st.session_state.state

    if agent_type == "DQN":
        action = agent.act(state)
    else:
        action, _ = agent.select_action(state, temperature=0.8)

    next_state, reward, done, _ = env.step(action)

    st.write("\u2705 Recommended Article:", news_titles.get(env.candidate_ids[action], 'Unknown Title'))
    st.write("\U0001F3C6 Reward (click):", reward)

    st.session_state.state = env.reset()
    st.session_state.candidate_ids = env.candidate_ids

if st.button("🔄 Reset Environment"):
    st.session_state.state = env.reset()
    st.session_state.candidate_ids = env.candidate_ids

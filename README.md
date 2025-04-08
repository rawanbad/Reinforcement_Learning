<h1 align="center"> Reinforcement Learning for Decision making in Stochastic Environment </h1>
<p align="center">
  Rawan Badarneh, Mohammed Abu Al Heja
  <p align="center">
    Technion
  </p>
</p>

### Description
This project implements two types of agents using reinforcement learning algorithms to solve decision-making in a stochastic environment:

1. **Optimal Agent**: Designed for smaller state spaces, this agent aims to act optimally to achieve the highest possible reward by maximizing efficiency in decision-making.

2. **Non-Optimal Agent**: Suited for larger state spaces, this agent utilizes Q-learning to accumulate the greatest number of points (rewards) possible. It is built to consistently achieve a positive overall reward, even in complex scenarios.

These implementations demonstrate the adaptability and scalability of reinforcement learning techniques across different environmental complexities.

## 🌟 Key Features
- **Optimal Agent**:
  - **State Space Representation**: The state space for the Optimal Agent is explicitly defined to facilitate precise decision-making.
  - **Value Iteration**: Utilizes value iteration to determine the optimal policy, ensuring the agent acts in a way that maximizes the reward in smaller state spaces.

- **Non-Optimal Agent**:
  - **Feature-Based State Representation**: Instead of a full state representation, this agent uses a feature-based approach to efficiently manage larger state spaces.
  - **Q-Learning**: Employs Q-learning, a form of reinforcement learning that allows the agent to learn the value of an action in a particular state. This approach helps the agent learn to accumulate points effectively and consistently achieve positive rewards.

 

## 🚀 Getting Started

### Prerequisites
- Python 3.11 
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/samarsamara/RealEstateAI.git
cd RealEstateAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create ".env" file that contains your OpenAI API key:
   ```
   API_KEY=<your_api_key>
   ``` 

### Running the Application

Launch the application:
```bash
python RealEstateAI.py
```

## 📦 Project Structure
```
RealEstateAI/
│── examples/
│   ├── example1.txt
│   ├── example2.txt
│   ├── example3.txt
│   └── example4.txt
├── Data/
│   └── los_angeles.csv
├── Code/
│   ├── RealEstateAI.py      # Main application script
│   ├── create_VDBs.py       # create the vector data base
│   ├── create_embeddings.py       # create the embedding for each data raw
│   ├── exploring_agent.py      # Agentic RAG
│   ├── interacting_agent.py    # conversation manager 
│   └── recommendation_agent.py       # return the final recommendation for the user
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## 💻 Technical Details

### Dependencies
```
qdrant-client
tiktoken
numpy
pandas
torch
transformers
faiss-cpu
sentence-transformers
tqdm
langchain_openai
json
os
langchain

```



## 🎯 Example Usage

1. Start the application.
2. Enter your preferences in natural language:
   - "I'm looking for a 3-bedroom house in Beverly Hills with budget 1m"
   - "I want a condo with 2-bedroom and one bathroom apartment in los angeles with a budget of $500K"
3. View personalized recommendations with property descriptions.



## App Architecture:

![Alt Text](https://github.com/samarsamara/RealEstateAI/blob/main/Image20250308195141.jpg)



## Notes :
1. Our AI agent exclusively **searches for properties in Los Angeles**, requiring users to input their **budget, number of bedrooms, and number of bathrooms** as mandatory criteria.
2. Link for the embedded data : https://drive.google.com/drive/folders/1bsPUZ-zUZCfAebIkm2s_D-fPxOXQSVx9?usp=sharing.
   

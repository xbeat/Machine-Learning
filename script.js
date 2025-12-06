document.addEventListener('DOMContentLoaded', () => {
    const content = [
        {
            id: 1,
            title: 'Agent Directory: The Superhero Database',
            explanation: `Imagine a giant database of superheroes, each with their own special powers. This is the Agent Directory. It's a place where our AI agents can register their skills and find other agents to team up with. Just like a superhero might search for a partner with super strength, an AI agent can search for another agent with "natural language processing" skills.`,
            code: `// Import necessary libraries
import chromadb
from chromadb.config import Settings
import uuid

// Define the AgentDirectory class
class AgentDirectory:
    def __init__(self):
        // Initialize the ChromaDB client to store data
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="agent_directory"
        ))
        // Create a collection to store agents
        self.collection = self.client.create_collection("agents")

    // Method to register a new agent
    def register_agent(self, capabilities, metadata):
        // Generate a unique ID for the agent
        agent_id = str(uuid.uuid4())
        // Add the agent's data to the collection
        self.collection.add(
            documents=[str(capabilities)],
            metadatas=[metadata],
            ids=[agent_id]
        )
        return agent_id

    // Method to find agents with specific capabilities
    def find_agents(self, query, n_results=5):
        // Query the collection for similar agents
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

// Example of how to use the AgentDirectory
directory = AgentDirectory()
agent_id = directory.register_agent(
    capabilities=["natural language processing", "data analysis"],
    metadata={"name": "AnalyticsAgent", "version": "1.0"}
)`
        },
        {
            id: 2,
            title: 'Discovery Protocol: The Rules of Teaming Up',
            explanation: 'Think of this as the set of rules superheroes follow to form a team. It helps them find the right partners for a mission. For example, a mission might need a "primary" skill like flying and a "secondary" skill like super hearing. The Discovery Protocol helps find the best agent for each role.',
            code: `// Import necessary libraries
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

// Define the AgentProfile data class
@dataclass
class AgentProfile:
    id: str
    capabilities: List[str]
    metadata: Dict

// Define the DiscoveryProtocol class
class DiscoveryProtocol:
    def __init__(self, directory):
        self.directory = directory
        // Define weights for different capability priorities
        self.capability_weights = {
            "primary": 1.0,
            "secondary": 0.5
        }

    // Method to find collaborators for a mission
    def find_collaborators(self, requirements, constraints=None):
        // Build a weighted query based on the requirements
        weighted_query = self._build_weighted_query(requirements)
        // Find candidate agents using the directory
        candidates = self.directory.find_agents(weighted_query)

        // Apply any constraints to filter the candidates
        if constraints:
            candidates = self._apply_constraints(candidates, constraints)

        // Rank the candidates to find the best fit
        return self._rank_candidates(candidates)

    // Method to build the weighted query
    def _build_weighted_query(self, requirements):
        query = ""
        for cap, priority in requirements.items():
            weight = self.capability_weights.get(priority, 0.3)
            query += f"{cap} * {weight} + "
        return query.rstrip(" + ")

// Example of how to use the DiscoveryProtocol
protocol = DiscoveryProtocol(directory)
collaborators = protocol.find_collaborators({
    "data analysis": "primary",
    "visualization": "secondary"
})`
        },
        {
            id: 3,
            title: 'Collaborative Planning: Creating a Battle Plan',
            explanation: 'Once a team of superheroes is assembled, they need a plan. The Collaborative Planning Engine is like the team leader who breaks down a big mission into smaller tasks and assigns them to the right agents. This ensures that everyone works together smoothly.',
            code: `// Import necessary libraries
from typing import Optional
import asyncio

// Define the CollaborativePlanner class
class CollaborativePlanner:
    def __init__(self, discovery_protocol):
        self.discovery = discovery_protocol
        self.task_queue = asyncio.Queue()

    // Method to create an execution plan for a task
    async def create_execution_plan(self, task_description):
        // Break down the task into smaller subtasks
        subtasks = self._decompose_task(task_description)

        execution_plan = []
        for subtask in subtasks:
            // Find agents that are capable of performing the subtask
            agents = self.discovery.find_collaborators(subtask.requirements)

            // Allocate the subtask to the best-fit agent
            allocation = await self._allocate_subtask(subtask, agents)
            execution_plan.append(allocation)

        return self._optimize_plan(execution_plan)

    // Method to decompose a task
    def _decompose_task(self, task):
        // This is where the logic for breaking down the task would go
        subtasks = []
        // ...
        return subtasks

    // Method to allocate a subtask to an agent
    async def _allocate_subtask(self, subtask, agents):
        // This is where the logic for allocating the subtask would go
        return {"subtask": subtask, "agent": agents[0]}`
        },
        {
            id: 4,
            title: 'Secure Transactions: The Secret Handshake',
            explanation: 'When superheroes exchange important information, they need to do it securely. This is like a secret handshake that only they know. The Secure Transaction Framework ensures that when agents exchange data, it is kept safe from prying eyes and that the data is not tampered with.',
            code: `// Import necessary libraries
import hashlib
from datetime import datetime
from typing import Any

// Define the SecureTransaction class
class SecureTransaction:
    def __init__(self):
        self.ledger = {}
        self.pending = {}

    // Method to initiate a new transaction
    def initiate_transaction(self,
                           sender_id: str,
                           receiver_id: str,
                           payload: Any) -> str:
        // Create a unique ID for the transaction
        transaction_id = self._generate_transaction_id(sender_id, receiver_id)

        // Create a hash of the payload for verification
        payload_hash = hashlib.sha256(str(payload).encode()).hexdigest()

        // Store the transaction in the pending list
        self.pending[transaction_id] = {
            'sender': sender_id,
            'receiver': receiver_id,
            'payload_hash': payload_hash,
            'status': 'pending',
            'timestamp': datetime.utcnow()
        }

        return transaction_id

    // Method to commit a transaction
    def commit_transaction(self, transaction_id: str, verification_hash: str):
        if transaction_id not in self.pending:
            raise ValueError("Invalid transaction ID")

        transaction = self.pending[transaction_id]
        // Check if the verification hash matches the payload hash
        if verification_hash == transaction['payload_hash']:
            // If it matches, move the transaction to the ledger
            self.ledger[transaction_id] = {
                **transaction,
                'status': 'completed'
            }
            del self.pending[transaction_id]
            return True
        return False`
        },
        {
            id: 5,
            title: 'Communication Protocol: The Superhero Hotline',
            explanation: 'Superheroes need a way to talk to each other. This is their special hotline. The Communication Protocol is a set of rules that allows agents to send and receive messages, just like how we use a phone to talk to our friends.',
            code: `// Import necessary libraries
import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Callable

// Define the MessageType enum
class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    TASK = "task"
    STATUS = "status"

// Define the Message data class
@dataclass
class Message:
    sender: str
    recipient: str
    msg_type: MessageType
    content: Dict[str, Any]
    correlation_id: str

// Define the CommunicationBus class
class CommunicationBus:
    def __init__(self):
        self.subscribers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()

    // Method to publish a message
    async def publish(self, message: Message):
        await self.message_queue.put(message)
        if message.recipient in self.subscribers:
            await self.subscribers[message.recipient](message)

    // Method to subscribe to messages
    def subscribe(self, agent_id: str, callback: Callable):
        self.subscribers[agent_id] = callback

    // Method to start the communication bus
    async def start(self):
        while True:
            message = await self.message_queue.get()
            // ... message processing logic ...
            self.message_queue.task_done()

// Example of how to use the CommunicationBus
async def agent_callback(message):
    print(f"Agent received: {message.content}")

comm_bus = CommunicationBus()
comm_bus.subscribe("agent1", agent_callback)`
        },
        {
            id: 6,
            title: 'Knowledge Base: The Superhero Library',
            explanation: \`Superheroes learn from their experiences. They keep a library of all their knowledge. The Knowledge Base is a place where agents can store what they've learned, so they can use that information in the future. It's like a super-brain for the entire agent team.\`,
            code: \`// Import necessary libraries
import numpy as np
from typing import List, Dict, Optional
import pickle

// Define the KnowledgeBase class
class KnowledgeBase:
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.knowledge_vectors = {}
        self.knowledge_metadata = {}

    // Method to store knowledge
    def store_knowledge(self,
                       key: str,
                       content: Dict,
                       embedding: np.ndarray,
                       metadata: Optional[Dict] = None):
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}")

        self.knowledge_vectors[key] = embedding
        self.knowledge_metadata[key] = {
            'content': content,
            'metadata': metadata or {},
            'access_count': 0
        }

    // Method to query knowledge
    def query_knowledge(self,
                       query_embedding: np.ndarray,
                       top_k: int = 5) -> List[Dict]:
        similarities = {}
        for key, vector in self.knowledge_vectors.items():
            sim = self._cosine_similarity(query_embedding, vector)
            similarities[key] = sim

        // Get top-k similar entries
        sorted_keys = sorted(similarities.items(),
                           key=lambda x: x[1],
                           reverse=True)[:top_k]

        results = []
        for key, sim in sorted_keys:
            self.knowledge_metadata[key]['access_count'] += 1
            results.append({
                'key': key,
                'content': self.knowledge_metadata[key]['content'],
                'similarity': sim
            })

        return results

    // Method to calculate cosine similarity
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\`
        }
    ];

    const main = document.getElementById('content');
    const nav = document.querySelector('nav ul');

    content.forEach(slide => {
        const slideElement = document.createElement('div');
        slideElement.className = 'slide';
        slideElement.id = \`slide-\${slide.id}\`;
        slideElement.innerHTML = \`
            <h2>\${slide.title}</h2>
            <p>\${slide.explanation}</p>
            <div class="simulation" id="sim-\${slide.id}">
                <!-- Simulation will go here -->
            </div>
            <pre><code>\${slide.code}</code></pre>
        \`;
        main.appendChild(slideElement);

        const navLink = document.createElement('li');
        navLink.innerHTML = \`<a href="#slide-\${slide.id}">\${slide.title.split(':')[0]}</a>\`;
        nav.appendChild(navLink);
    });

    // Simulation for Agent Directory
    const sim1 = document.getElementById('sim-1');
    const agents = [
        { name: 'Agent A', skills: ['flying', 'super strength'] },
        { name: 'Agent B', skills: ['invisibility', 'super speed'] },
        { name: 'Agent C', skills: ['telepathy', 'super strength'] }
    ];

    sim1.innerHTML = \`
        <h3>Superhero Agent Directory</h3>
        <input type="text" id="skill-search" placeholder="Enter a skill to search for">
        <button id="search-btn">Search</button>
        <ul id="agent-list"></ul>
    \`;

    const searchBtn = document.getElementById('search-btn');
    const skillSearch = document.getElementById('skill-search');
    const agentList = document.getElementById('agent-list');

    searchBtn.addEventListener('click', () => {
        const query = skillSearch.value.toLowerCase();
        agentList.innerHTML = '';
        const results = agents.filter(agent => agent.skills.includes(query));
        if (results.length > 0) {
            results.forEach(agent => {
                const li = document.createElement('li');
                li.textContent = \`\${agent.name} has skills: \${agent.skills.join(', ')}\`;
                agentList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No agents found with that skill.';
            agentList.appendChild(li);
        }
    });

    // Simulation for Discovery Protocol
    const sim2 = document.getElementById('sim-2');
    sim2.innerHTML = \`
        <h3>Mission Team Builder</h3>
        <input type="text" id="primary-skill" placeholder="Enter primary skill">
        <input type="text" id="secondary-skill" placeholder="Enter secondary skill">
        <button id="find-team-btn">Find Team</button>
        <p id="team-result"></p>
    \`;

    const findTeamBtn = document.getElementById('find-team-btn');
    const primarySkill = document.getElementById('primary-skill');
    const secondarySkill = document.getElementById('secondary-skill');
    const teamResult = document.getElementById('team-result');

    findTeamBtn.addEventListener('click', () => {
        const primary = primarySkill.value.toLowerCase();
        const secondary = secondarySkill.value.toLowerCase();
        let bestAgent = null;
        let bestScore = -1;

        agents.forEach(agent => {
            let score = 0;
            if (agent.skills.includes(primary)) {
                score += 1.0;
            }
            if (agent.skills.includes(secondary)) {
                score += 0.5;
            }
            if (score > bestScore) {
                bestScore = score;
                bestAgent = agent;
            }
        });

        if (bestAgent) {
            teamResult.textContent = \`Best agent for the mission is \${bestAgent.name} with a score of \${bestScore}\`;
        } else {
            teamResult.textContent = 'No suitable agent found for this mission.';
        }
    });

    // Simulation for Collaborative Planning
    const sim3 = document.getElementById('sim-3');
    sim3.innerHTML = \`
        <h3>Mission Planner</h3>
        <input type="text" id="mission-desc" placeholder="Enter mission description">
        <button id="plan-mission-btn">Plan Mission</button>
        <ul id="mission-plan"></ul>
    \`;

    const planMissionBtn = document.getElementById('plan-mission-btn');
    const missionDesc = document.getElementById('mission-desc');
    const missionPlan = document.getElementById('mission-plan');

    planMissionBtn.addEventListener('click', () => {
        const description = missionDesc.value.toLowerCase();
        missionPlan.innerHTML = '';

        // Simple task decomposition
        const subtasks = description.split('and');

        subtasks.forEach(task => {
            let bestAgent = null;
            let bestScore = -1;

            agents.forEach(agent => {
                let score = 0;
                agent.skills.forEach(skill => {
                    if (task.includes(skill)) {
                        score += 1;
                    }
                });
                if (score > bestScore) {
                    bestScore = score;
                    bestAgent = agent;
                }
            });

            const li = document.createElement('li');
            if (bestAgent) {
                li.textContent = \`Task: "\${task.trim()}" assigned to \${bestAgent.name}\`;
            } else {
                li.textContent = \`Task: "\${task.trim()}" - No suitable agent found\`;
            }
            missionPlan.appendChild(li);
        });
    });

    // Simulation for Secure Transactions
    const sim4 = document.getElementById('sim-4');
    sim4.innerHTML = \`
        <h3>Secure Message Sender</h3>
        <input type="text" id="message-to-send" placeholder="Enter a message">
        <button id="send-message-btn">Send Secure Message</button>
        <p id="message-status"></p>
    \`;

    const sendMessageBtn = document.getElementById('send-message-btn');
    const messageToSend = document.getElementById('message-to-send');
    const messageStatus = document.getElementById('message-status');

    sendMessageBtn.addEventListener('click', () => {
        const message = messageToSend.value;
        const hash = simpleHash(message);
        messageStatus.textContent = \`Message sent with hash: \${hash}\`;
    });

    function simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = (hash << 5) - hash + char;
            hash &= hash; // Convert to 32bit integer
        }
        return hash;
    }

    // Simulation for Communication Protocol
    const sim5 = document.getElementById('sim-5');
    sim5.innerHTML = \`
        <h3>Superhero Hotline</h3>
        <select id="agent-selector"></select>
        <input type="text" id="agent-message" placeholder="Enter a message">
        <button id="send-agent-message-btn">Send Message</button>
        <ul id="message-log"></ul>
    \`;

    const agentSelector = document.getElementById('agent-selector');
    const agentMessage = document.getElementById('agent-message');
    const sendAgentMessageBtn = document.getElementById('send-agent-message-btn');
    const messageLog = document.getElementById('message-log');

    agents.forEach(agent => {
        const option = document.createElement('option');
        option.value = agent.name;
        option.textContent = agent.name;
        agentSelector.appendChild(option);
    });

    sendAgentMessageBtn.addEventListener('click', () => {
        const recipient = agentSelector.value;
        const message = agentMessage.value;
        const li = document.createElement('li');
        li.textContent = \`Message to \${recipient}: \${message}\`;
        messageLog.appendChild(li);
    });

    // Simulation for Knowledge Base
    const sim6 = document.getElementById('sim-6');
    const knowledgeBase = {};

    sim6.innerHTML = \`
        <h3>Superhero Library</h3>
        <input type="text" id="knowledge-key" placeholder="Enter knowledge key">
        <input type="text" id="knowledge-value" placeholder="Enter knowledge value">
        <button id="store-knowledge-btn">Store Knowledge</button>
        <input type="text" id="knowledge-query" placeholder="Enter knowledge to query">
        <button id="query-knowledge-btn">Query Knowledge</button>
        <p id="knowledge-result"></p>
    \`;

    const storeKnowledgeBtn = document.getElementById('store-knowledge-btn');
    const knowledgeKey = document.getElementById('knowledge-key');
    const knowledgeValue = document.getElementById('knowledge-value');
    const queryKnowledgeBtn = document.getElementById('query-knowledge-btn');
    const knowledgeQuery = document.getElementById('knowledge-query');
    const knowledgeResult = document.getElementById('knowledge-result');

    storeKnowledgeBtn.addEventListener('click', () => {
        const key = knowledgeKey.value;
        const value = knowledgeValue.value;
        knowledgeBase[key] = value;
        knowledgeKey.value = '';
        knowledgeValue.value = '';
    });

    queryKnowledgeBtn.addEventListener('click', () => {
        const query = knowledgeQuery.value;
        if (knowledgeBase[query]) {
            knowledgeResult.textContent = \`Knowledge found: \${knowledgeBase[query]}\`;
        } else {
            knowledgeResult.textContent = 'No knowledge found for that query.';
        }
    });
});

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
class Node:
    """
    Base class for all nodes in the network.
    """
    def __init__(self, energy_capacity, failure_probability=0.3, demand_change_probability=0.2, repair_probability=0.5):
        """
        Initialize the node with energy capacity, failure probability, demand change probability, and repair probability.
        """
        self.energy_capacity = energy_capacity
        self.energy_level = 0
        self.energy_history = []  # Store energy reception history
        self.energy_storage = 0  # Store received energy
        self.failure_probability = failure_probability  # Probability of node failure
        self.demand_change_probability = demand_change_probability  # Probability of change in energy demand
        self.repair_probability = repair_probability  # Probability of node repair
        self.failure_mode = "random"  # Node failure mode (random, periodic, etc.)

    def receive_energy(self, energy_signal):
        """
        Receive energy from the hub or other nodes.
        """
        optimized_energy = self.optimize_energy_reception(energy_signal)
        self.energy_level += optimized_energy
        self.energy_history.append(optimized_energy)  # Update energy history
        self.energy_storage += optimized_energy  # Store received energy

    def optimize_energy_reception(self, energy_signal):
        """
        Optimize energy reception using a moving average filter.
        """
        window_size = 5
        if len(self.energy_history) < window_size:
            return energy_signal
        else:
            avg_energy = np.mean(self.energy_history[-window_size:])
            return energy_signal * (1 + (avg_energy / self.energy_capacity))

    def predict_energy(self, state):
        """
        Predict energy demand using a moving average prediction model.
        """
        window_size = 5
        if len(self.energy_history) < window_size:
            return state * 0.8
        else:
            avg_energy = np.mean(self.energy_history[-window_size:])
            return avg_energy * 0.9 + state * 0.1

    def dedicate_energy(self, other_node):
        """
        Dedicate energy to other system networks.
        """
        energy_transfer = self.predict_energy(self.energy_level)
        other_node.receive_energy(energy_transfer)

    def check_failure(self):
        """
        Check if the node has failed.
        """
        if self.failure_mode == "random":
            if random.random() < self.failure_probability:
                self.energy_level = 0
                self.energy_storage = 0
                return True
        elif self.failure_mode == "periodic":
            if random.random() < 0.1:  # 10% chance of failure every 10 steps
                self.energy_level = 0
                self.energy_storage = 0
                return True
        return False

    def update_demand(self):
        """
        Update energy demand.
        """
        if random.random() < self.demand_change_probability:
            self.energy_capacity *= random.uniform(0.5, 1.5)
            return True
        else:
            return False

    def repair(self):
        """
        Repair the node.
        """
        if random.random() < self.repair_probability:
            self.energy_level = self.energy_capacity
            self.energy_storage = self.energy_capacity
            return True
        else:
            return False

    def strong_repair(self):
        """
        Strong repair: restore node to full capacity and reset failure probability.
        """
        self.energy_level = self.energy_capacity
        self.energy_storage = self.energy_capacity
        self.failure_probability = 0.01  # Reset failure probability to a low value
        return True

    def trade_energy(self, other_node):
        """
        Trade energy with another node.
        """
        energy_transfer = min(self.energy_level, other_node.energy_capacity - other_node.energy_level)
        self.energy_level -= energy_transfer
        other_node.energy_level += energy_transfer
        return energy_transfer

class Hub(Node):
    """
    Central hub of the network.
    """
    def __init__(self, energy_capacity):
        """
        Initialize the hub with energy capacity.
        """
        super().__init__(energy_capacity)
        self.nodes = []  # List of nodes connected to the hub

    def add_node(self, node):
        """
        Add a node to the hub.
        """
        self.nodes.append(node)

    def distribute_energy(self):
        """
        Distribute energy to connected nodes.
        """
        for node in self.nodes:
            energy_signal = random.random()
            node.receive_energy(energy_signal)
            self.energy_level = max(random.random(), self.energy_level - energy_signal)  # Update hub's energy level
            self.energy_storage = max(random.random(), self.energy_storage - energy_signal)  # Update hub's energy storage

class ElectricityNode(Node):
    """
    Electricity node.
    """
    def __init__(self, energy_level, resource_level):
        """
        Initialize the electricity node with energy level and resource level.
        """
        super().__init__(energy_level)
        self.resource_level = resource_level
        self.energy_history = []
        self.energy_storage = 0  # Initialize energy storage for ElectricityNode

    def update_demand(self):
        """
        Update energy demand based on a more complex model.
        """
        if random.random() < self.demand_change_probability:
            self.energy_capacity *= random.uniform(0.5, 1.5)
            self.resource_level *= random.uniform(0.5, 1.5)
            return True
        else:
            return False

class WaterNode(Node):
    """
    Water node.
    """
    def __init__(self, energy_level, resource_level):
        """
        Initialize the water node with energy level and resource level.
        """
        super().__init__(energy_level)
        self.quality = 0.5  # default water quality
        self.resource_level = resource_level
        self.energy_history = []
        self.energy_storage = 0  # Initialize energy storage for WaterNode

    def update_demand(self):
        """
        Update energy demand based on a more complex model.
        """
        if random.random() < self.demand_change_probability:
            self.energy_capacity *= random.uniform(0.5, 1.5)
            self.resource_level *= random.uniform(0.5, 1.5)
            self.quality *= random.uniform(0.5, 1.5)
            return True
        else:
            return False

class GasNode(Node):
    """
    Gas node.
    """
    def __init__(self, energy_level, resource_level):
        """
        Initialize the gas node with energy level and resource level.
        """
        super().__init__(energy_level)
        self.resource_level = resource_level
        self.energy_history = []
        self.energy_storage = 0  # Initialize energy storage for GasNode

    def update_demand(self):
        """
        Update energy demand based on a more complex model.
        """
        if random.random() < self.demand_change_probability:
            self.energy_capacity *= random.uniform(0.5, 1.5)
            self.resource_level *= random.uniform(0.5, 1.5)
            return True
        else:
            return False

class TransportationNode(Node):
    """
    Transportation node.
    """
    def __init__(self, energy_capacity, speed, capacity, resource_level):
        """
        Initialize the transportation node with energy capacity, speed, capacity, and resource level.
        """
        super().__init__(energy_capacity)
        self.speed = speed
        self.capacity = capacity
        self.route = "default route"  # default transportation route
        self.resource_level = resource_level
        self.energy_history = []
        self.energy_storage = 0  # Initialize energy storage for TransportationNode

    def update_demand(self):
        """
        Update energy demand based on a more complex model.
        """
        if random.random() < self.demand_change_probability:
            self.energy_capacity *= random.uniform(0.5, 1.5)
            self.resource_level *= random.uniform(0.5, 1.5)
            self.speed *= random.uniform(0.5, 1.5)
            self.capacity *= random.uniform(0.5, 1.5)
            return True
        else:
            return False

    def trade_energy(self, other_node):
        """
        Trade energy with another node.
        """
        energy_transfer = min(self.energy_level, other_node.energy_capacity - other_node.energy_level)
        self.energy_level -= energy_transfer
        other_node.energy_level += energy_transfer
        return energy_transfer

class AuroraNetwork:
    """
    Aurora network.
    """
    def __init__(self, num_nodes):
        """
        Initialize the Aurora network with a specified number of nodes.
        """
        self.nodes = []
        self.hub = Hub(1000)  # Create a hub with 1000 energy capacity
        locations = ["City A", "City B", "City C", "City D", "City E"]
        weathers = ["Sunny", "Cloudy", "Rainy", "Windy", "Snowy"]
        for _ in range(num_nodes):
            node_type = random.choice(["Electricity", "Water", "Gas", "Transportation"])
            location = random.choice(locations)
            weather = random.choice(weathers)
            if node_type == "Electricity":
                node = ElectricityNode(100, 120)
            elif node_type == "Water":
                node = WaterNode(100, 10)
            elif node_type == "Gas":
                node = GasNode(100, 5)
            else:
                node = TransportationNode(100, 60, 50, 100)
            self.hub.add_node(node)  # Add node to the hub
            self.nodes.append(node)

    def simulate(self, num_steps):
        """
        Simulate the Aurora network for a specified number of steps.
        """
        for _ in range(num_steps):
            self.hub.distribute_energy()

            # Simulate energy reception and prediction at each node
            for node in self.nodes:
                energy_signal = random.random()
                node.receive_energy(energy_signal)
                node.predict_energy(node.energy_level)

                # Check for node failure
                if node.check_failure():
                    print(f"Node {node} has failed.")

                # Update energy demand
                if node.update_demand():
                    print(f"Energy demand at node {node} has changed.")

                # Strong repair: restore node to full capacity and reset failure probability
                if node.strong_repair():
                    print(f"Node {node} has been strongly repaired.")

            # Simulate energy trading between nodes
            for i in range(len(self.nodes)):
                for j in range(i+1, len(self.nodes)):
                    if hasattr(self.nodes[i], 'trade_energy'):
                        energy_transfer = self.nodes[i].trade_energy(self.nodes[j])
                        print(f"Node {self.nodes[i]} traded {energy_transfer} energy with node {self.nodes[j]}")

            # Simulate energy dissipation
            for node in self.nodes:
                node.energy_level *= 0.9  # 10% energy dissipation
class AuroraBorealis:
    def __init__(self):
        self.energy_level = 1000  # Initial energy level of the Aurora Borealis
        self.energy_fluctuation = 0.1  # Fluctuation in energy level (10%)

    def simulate(self):
        # Simulate the energy fluctuation of the Aurora Borealis
        self.energy_level *= (1 + np.random.uniform(-self.energy_fluctuation, self.energy_fluctuation))

class CosmicEnergyExtractor:
    def __init__(self, efficiency):
        self.efficiency = efficiency  # Efficiency of the cosmic energy extractor

    def extract_energy(self, aurora_energy):
        # Extract cosmic energy from the Aurora Borealis
        return aurora_energy * self.efficiency

class CosmicWaveNotificationSystem:
    def __init__(self, pattern):
        self.pattern = pattern  # Notification pattern

    def detect_cosmic_wave(self, energy_level):
        # Detect cosmic wave pattern
        wave_pattern = np.array([1, 0, 1, 1, 0, 1])  # Example pattern
        if np.allclose(energy_level % 10, wave_pattern):
            return True
        else:
            return False

    def send_notification(self):
        # Send a notification
        print("Aurora Event Detected through Cosmic Waves!")

class Simulation:
    def __init__(self, num_steps):
        self.num_steps = num_steps
        self.aurora = AuroraBorealis()
        self.extractor = CosmicEnergyExtractor(0.2)  # 20% efficiency
        self.cosmic_energy = 0
        self.notification_system = CosmicWaveNotificationSystem(np.array([1, 0, 1, 1, 0, 1]))  # Notification pattern

    def run(self):
        for _ in range(self.num_steps):
            self.aurora.simulate()
            extracted_energy = self.extractor.extract_energy(self.aurora.energy_level)
            self.cosmic_energy += extracted_energy
            print(f"Step {_+1}: Aurora Energy Level = {self.aurora.energy_level:.2f}, Extracted Energy = {extracted_energy:.2f}, Total Cosmic Energy = {self.cosmic_energy:.2f}")

            # Detect cosmic wave pattern
            if self.notification_system.detect_cosmic_wave(self.aurora.energy_level):
                self.notification_system.send_notification()
# Create an Aurora network with 5 nodes
num_nodes = 5
aurora_network = AuroraNetwork(num_nodes)
# Simulate the network for 10 steps
aurora_network.simulate(10)


# Print the network status
for node in aurora_network.nodes:
    print(f"Node {node}: {node.__class__.__name__} - Energy Level: {str(node.energy_level)}, Resource Level: {str(node.resource_level)}, Energy Storage: {str(node.energy_storage)}")

#
print(f"Hub: Energy Level: {str(aurora_network.hub.energy_level)}, Energy Storage: {str(aurora_network.hub.energy_storage)}")
num_steps = 10
energy_levels = np.zeros((num_nodes, num_steps))
resource_levels = np.zeros((num_nodes, num_steps))
energy_storages = np.zeros((num_nodes, num_steps))
hub_energy_levels = np.zeros(num_steps)
hub_energy_storages = np.zeros(num_steps)

for i in range(num_steps):
    aurora_network.simulate(1)
    for j, node in enumerate(aurora_network.nodes):
        energy_levels[j, i] = node.energy_level
        resource_levels[j, i] = node.resource_level
        energy_storages[j, i] = node.energy_storage
    hub_energy_levels[i] = aurora_network.hub.energy_level
    hub_energy_storages[i] = aurora_network.hub.energy_storage

# Plot energy levels over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=energy_levels.T)
plt.xlabel('Time Step')
plt.ylabel('Energy Level')
plt.title('Energy Levels Over Time')
plt.legend([f'Node {i+1}' for i in range(num_nodes)])
plt.show()

# Plot resource levels over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=resource_levels.T)
plt.xlabel('Time Step')
plt.ylabel('Resource Level')
plt.title('Resource Levels Over Time')
plt.legend([f'Node {i+1}' for i in range(num_nodes)])
plt.show()

# Plot energy storages over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=energy_storages.T)
plt.xlabel('Time Step')
plt.ylabel('Energy Storage')
plt.title('Energy Storages Over Time')
plt.legend([f'Node {i+1}' for i in range(num_nodes)])
plt.show()

# Plot hub energy levels over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=hub_energy_levels)
plt.xlabel('Time Step')
plt.ylabel('Energy Level')
plt.title('Hub Energy Levels Over Time')
plt.show()

# Plot hub energy storages over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=hub_energy_storages)
plt.xlabel('Time Step')
plt.ylabel('Energy Storage')
plt.title('Hub Energy Storages Over Time')
plt.show()
simulation = Simulation(10)
simulation.run()
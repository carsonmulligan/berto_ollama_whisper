import networkx as nx
import matplotlib.pyplot as plt

# Initialize the graph
G = nx.DiGraph()

# Define nodes
nodes = [
    "Python Packaging User Guide",
    "Build Backend",
    "Build Frontend",
    "Built Distribution",
    "Built Metadata",
    "Core Metadata",
    "Core Metadata Field",
    "Distribution Archive",
    "Distribution Package",
    "Egg",
    "Extension Module",
    "Known Good Set (KGS)",
    "Import Package",
    "Installed Project",
    "Module",
    "Pure Module",
    "Package Index",
    "Python Package Index (PyPI)",
    "Per Project Index",
    "Project",
    "Project Root Directory",
    "Project Source Tree",
    "Project Source Metadata",
    "Pyproject Metadata",
    "Pyproject Metadata Key",
    "Pyproject Metadata Subkey",
    "Python Packaging Authority (PyPA)",
    "pypi.org",
    "pyproject.toml",
    "Release",
    "Requirement",
    "Requirement Specifier",
    "Requirements File",
    "setup.py",
    "setup.cfg",
    "Source Archive",
    "Source Distribution (sdist)",
    "System Package",
    "Version Specifier",
    "Virtual Environment",
    "Wheel Format",
    "Wheel",
    "Wheel Project",
    "Working Set"
]

# Add nodes to the graph
G.add_nodes_from(nodes)

# Define edges (relationships)
edges = [
    ("Build Frontend", "Build Backend"),
    ("Build Backend", "Built Distribution"),
    ("Built Distribution", "Distribution Package"),
    ("Egg", "Built Distribution"),
    ("Extension Module", "Module"),
    ("Pure Module", "Module"),
    ("Module", "Import Package"),
    ("Project", "Build Backend"),
    ("Project", "Build Frontend"),
    ("Project", "Project Source Metadata"),
    ("Pyproject Metadata", "Project Source Metadata"),
    ("Pyproject Metadata Key", "Pyproject Metadata"),
    ("Pyproject Metadata Subkey", "Pyproject Metadata Key"),
    ("Python Packaging Authority (PyPA)", "Python Package Index (PyPI)"),
    ("Python Packaging Authority (PyPA)", "pyproject.toml"),
    ("Python Package Index (PyPI)", "Package Index"),
    ("pypi.org", "Python Package Index (PyPI)"),
    ("Project", "Release"),
    ("Release", "Distribution Package"),
    ("Requirement", "Requirement Specifier"),
    ("Requirement Specifier", "Requirements File"),
    ("setup.py", "Project"),
    ("setup.cfg", "Project"),
    ("Source Archive", "Source Distribution (sdist)"),
    ("Source Distribution (sdist)", "Built Distribution"),
    ("Virtual Environment", "Installed Project"),
    ("Wheel Format", "Built Distribution"),
    ("Wheel", "Wheel Format"),
    ("Wheel Project", "Wheel Format"),
    ("Working Set", "Installed Project"),
    ("Known Good Set (KGS)", "Working Set"),
    ("Distribution Package", "Distribution Archive"),
    ("Built Metadata", "Distribution Package"),
    ("Core Metadata", "Built Metadata"),
    ("Core Metadata Field", "Core Metadata"),
    ("Project Source Tree", "Project"),
    ("Project Root Directory", "Project"),
    ("Project Source Metadata", "Project Source Tree"),
    ("Project Source Metadata", "pyproject.toml"),
    ("Virtual Environment", "Project"),
    ("System Package", "Distribution Package")
]

# Add edges to the graph
G.add_edges_from(edges)

# Define positions using spring layout for better visualization
pos = nx.spring_layout(G, k=0.5, iterations=100)

# Draw the nodes
plt.figure(figsize=(20, 20))
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="lightblue")

# Draw the edges
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")

# Draw the labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

# Remove axes
plt.axis("off")
plt.title("Interactive Graph of Python Packaging Concepts", fontsize=20)
plt.show()

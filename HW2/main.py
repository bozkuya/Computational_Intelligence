#import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
# Parameters
#This part is changed with different parameters according to table.
#Bold parts are standart and one variable is changed in each experiment
number_of_individuals = 20 # number of individuals
number_of_genes = 50 # number of genes
num_generation = 10000 # number of generations
tm_size = 5 # tournament size
num_elites = 0.2 # fraction of elites
num_parents = 0.6 # fraction of parents
mutation_prob = 0.75 # mutation probability
mutation_type = "guided" #guided ungided selections
width, height = 256, 256 # dimension of image
max_radius = int(np.sqrt(width**2 + height**2) / 2)
# Reading source image
#Path for image
#
source_image = cv2.imread(r'C:\Users\yasin\Downloads\painting.png', 
cv2.COLOR_BGR2RGB)
source_image = cv2.resize(source_image, (width, height))
#gene classs
class Gene:
def __init__(self):
self.y = np.random.randint(0, width)
self.x = np.random.randint(0, height)
self.radius = np.random.randint(1, max_radius)
self.r = np.random.randint(0, 256)
self.g = np.random.randint(0, 256)
self.b = np.random.randint(0, 256)
self.a = np.random.rand()
def mutate(self):
if mutation_type == "guided":
self.x = np.clip(self.x + np.random.randint(-width//4, width//4), 
0, width)
self.y = np.clip(self.y + np.random.randint(-height//4, 
height//4), 0, height)
self.radius = np.clip(self.radius + np.random.randint(-10, 10), 1, 
width // 2)
self.r = np.clip(self.r + np.random.randint(-64, 64), 0, 256)

self.g = np.clip(self.g + np.random.randint(-64, 64), 0, 256)
self.b = np.clip(self.b + np.random.randint(-64, 64), 0, 256)
self.a = np.clip(self.a + np.random.uniform(-0.25, 0.25), 0, 1)
else: # unguided mutation
#directly use
self.__init__()
#individial class
class Individual:
def __init__(self, genes=None):
if genes is None:
self.genes = [Gene() for _ in range(number_of_genes)]
else:
self.genes = genes
self.fitness = self.evaluate()
def evaluate(self):
# omitted for brevity
def sort_genes(self):
self.genes.sort(key=lambda gene: gene.radius, reverse=True)
def evaluate(self):
image = np.ones((width, height, 3), np.uint8) * 255
for gene in self.genes:
overlay = image.copy()
cv2.circle(overlay, (gene.x, gene.y), gene.radius, (gene.b, 
gene.g, gene.r), -1)
image = cv2.addWeighted(overlay, gene.a, image, 1 - gene.a, 0)
self.image = image
self.fitness = -np.sum((source_image.astype("float") -
image.astype("float")) ** 2)
return self.fitness
#populatıon class
class Population:
def __init__(self):
self.individuals = [Individual() for _ in
range(number_of_individuals)]
def selection(self):
self.individuals.sort(key=lambda individual: individual.fitness, 
reverse=True)
next_generation = 
self.individuals[:int(num_elites*number_of_individuals)]
for _ in range(int(number_of_individuals -
num_elites*number_of_individuals)):
tournament = np.random.choice(self.individuals, tm_size)
tournament = sorted(tournament, key=lambda individual: 
individual.fitness, reverse=True)

next_generation.append(tournament[0])
self.individuals = next_generation
def crossover(self):
num_cross = int((number_of_individuals -
num_elites*number_of_individuals)/2)
for _ in range(num_cross):
# select parents
parents = 
random.sample(self.individuals[int(num_elites*number_of_individuals):int((num_
elites+num_parents)*number_of_individuals)], 2)
for i in range(2):
# create child
child_genes = [parents[np.random.randint(0, 2)].genes[j] for j
in range(number_of_genes)]
child = Individual(child_genes)
# add child to population
self.individuals.append(child)
def mutation(self):
for individual in
self.individuals[int(num_elites*number_of_individuals):]:
if np.random.rand() < mutation_prob:
gene = individual.genes[np.random.randint(0, number_of_genes)]
if mutation_type == "unguided":
gene.x = np.random.randint(0, width)
gene.y = np.random.randint(0, height)
gene.radius = np.random.randint(1, max_radius)
gene.r = np.random.randint(0, 256)
gene.g = np.random.randint(0, 256)
gene.b = np.random.randint(0, 256)
gene.a = np.random.rand()
elif mutation_type == "guided":
gene.x = min(max(0, gene.x + np.random.randint(-width//4, 
width//4)), width-1)
gene.y = min(max(0, gene.y + np.random.randint(-height//4, 
height//4)), height-1)
gene.radius = min(max(1, gene.radius + np.random.randint(-
10, 10)), max_radius)
gene.r = min(max(0, gene.r + np.random.randint(-64, 64)), 
255)
gene.g = min(max(0, gene.g + np.random.randint(-64, 64)), 
255)
gene.b = min(max(0, gene.b + np.random.randint(-64, 64)), 
255)
gene.a = min(max(0, gene.a + (np.random.rand()-0.5)/2), 1)
individual.fitness = individual.evaluate()

2304202
# Run genetic algorithm
population = Population()
best_individuals = []
best_fitnesses = []
for generation in range(num_generation):
population.selection()
population.crossover()
population.mutation()
population.individuals.sort(key=lambda individual: individual.fitness, 
reverse=True)
best_fitnesses.append(population.individuals[0].fitness)
# For each 1000 generation
if generation % 1000 == 0:
best_individuals.append(population.individuals[0])
print(f"Generation {generation}")
# Plot fitness
plt.plot(best_fitnesses)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()
# Display image as a result
for i, individual in enumerate(best_individuals):
plt.imshow(cv2.cvtColor(individual.image, cv2.COLOR_BGR2RGB))
plt.title(f"Generation {i*1000}")
plt.show()
#Yasincan B
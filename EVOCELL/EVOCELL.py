# ---   EvoCell   ---
# An evolutional approach to Neural Network architecture optimization on Breast
# cancer cell line classification.
import random
from Modules import Net, Evaluate

TORQUE_LIST  =  {"BR":1.0, "BI":1.0, "BB":1.0, "BC":1.0,
                 "RR":1.0, "RI":1.0, "RB":1.0, "RC":1.0,
                 "CR":1.0, "CI":1.0, "CB":1.0, "CC":1.0,
                 "IR":1.0, "II":1.0, "IB":1.0, "IC":1.0,
                 }

PRESENCE_RATIO =  {"Bot":0.0, "Res":0.0, "Invr":0.0, "Clru":0.0}

""" Individual/Chromosome maker block """

#Rank fitness
#Store results for generation
#Reproduce- crossover and mutation
#Prune population

def gene_creator(input_features):
    gene = []
    blocks = ["Res", "Invr", "Bot", "CrLu"]
    out_channels = [8]

    for pos in range(7):
        block = random.choice(blocks)
        out = random.choice(out_channels)

        if block == "Invr":
            stride = random.choice([1,2])
            if pos == 0:
                gene.append([block, input_features, out, 1, stride])
                continue
            gene.append([block, gene[-1][2], out, 6, stride])
            continue

        if pos == 0:
            gene.append([block, input_features, out])

        else: 
            """ Avoiding Residual blocks after Bottlenecks """
            if pos > 0 and gene[pos-1][0] == "Bot" and block == "Res":
                new_block = random.choice(["Invr", "Bot", "CrLu"])
                gene.append([new_block, gene[-1][2], out])
                continue
            gene.append([block, gene[-1][2], out])

    return gene

def gene_corrector(gene):

    #ResNet fix - identical outputs
    for pos in range(1,len(gene)):
        prev_block_output = gene[pos-1][2]
        block = gene[pos]

        if block[0] == 'Res':
            block[1] = prev_block_output
            block[2] = block[1]

    #Refix input output feature mappings
    for pos in range(1,len(gene)):
        block = gene[pos]
        prev_block = gene[pos-1]
        if block[0] == 'Res': continue
        if block[0] == 'Bot': continue
        if block[1] != prev_block[2]: 
            block[1] = prev_block[2]

    # Every bottleneck output is a 4th of input
    for pos in range(0,len(gene)):
        if gene[pos][0] == 'Bot':
            gene[pos][2] = gene[pos][1] // 4

    # Correct the input channels to blocks after bottleneck blocks
    #for pos in range(1,len(gene)):
     #   if gene[pos-1][0] == 'Bot':
      #      gene[pos][1] = gene[pos-1][2] * 4

    

    return gene

def make_chromosome(input_features):
    chromosome = {"Gene":gene_corrector(gene_creator(input_features)), "Torque_list":TORQUE_LIST, 
                  "Presence_ratios":PRESENCE_RATIO, "Score":0.0}
    return (chromosome)


parent_a = make_chromosome(8)
parent_b = make_chromosome(8)
print (parent_a["Gene"])
print (parent_b["Gene"])

""" Mutation and Crossover Auxiliary functions """
#Extract sets
def extract_sets(parent_gene):
    sets =  []
    for idx in range(len(parent_gene)-1):
        set_block_a = parent_gene[idx][0][0]
        set_block_b = parent_gene[idx+1][0][0]
        name_set = [set_block_a+set_block_b]
        sets.append(name_set)
    return sets

#Check for similar sets
def similar_set_checker(set1, set2):
    similar_listA = [group for group in set1 if group in set2]
    return similar_listA

#Finds average of similar torques
def torque_average(trq_list_a, trq_list_b, similarity_list):
    avg_torque = []
    for ele in similarity_list:
        prt_a_torque = trq_list_a[ele[0]]
        prt_b_torque = trq_list_b[ele[0]]

        avg_torque.append((prt_a_torque + prt_b_torque) // 2)
    return avg_torque

#Find max similar torque 
def max_torque(avgs_list, similarity_list):
    max_idx = avgs_list.index(max(avgs_list))
    return similarity_list[max_idx]

#Find max set torque in individual
def max_set_torque(parent):
    return max(parent["Torque_list"], key=parent["Torque_list"].get)

#Find min set torque in individual
def min_set_torque(parent):
    return min(parent["Torque_list"], key=parent["Torque_list"].get)

#Find max presence ratio block 
def max_model_block(parent):
    return max(parent["Presence_ratios"], key=parent["Presence_ratios"].get)

#Find min presence ratio block
def min_model_block(parent):
    return min(parent["Presence_ratios"], key=parent["Presence_ratios"].get)

""" CROSSOVER Auxiliary functions """
#Gets first half of offspring
def front_gene(parent_gene, highest_torque_set):
    max_set = highest_torque_set#highest_torque_set[0] 

    for idx in range(len(parent_gene)-1,-1,-1):
        #Searching for max torque set, through individuals blocks
        block = parent_gene[idx][0] #Actual index for current block
        prev_block = parent_gene[idx-1][0]

        idx_cut = random.randint(1,7)

        if prev_block[0] + block[0] == max_set[0] + max_set[1]: #Last block in set == Last block in max_set
            idx_cut = idx
        if idx == 1: break

    front_gene_res = [gene for gene in parent_gene if parent_gene.index(gene) <= idx_cut]
    return front_gene_res 

#Gets second half of offspring
def back_gene(parent_gene, highest_torque_set):
    max_set = highest_torque_set#highest_torque_set[0] 

    for idx in range(len(parent_gene)):

        if idx == len(parent_gene)-1: break

        block = parent_gene[idx][0] #Actual index for current block
        next_block = parent_gene[idx+1][0]

        idx_cut = random.randint(1,7)

        if block[0] + next_block[0] == max_set[0] + max_set[1]: #First block in set == First block in max_set
            idx_cut = idx
    

    back_gene_res = [gene for gene in parent_gene if parent_gene.index(gene) >= idx_cut]
    return back_gene_res 


print (parent_a["Torque_list"])
print (parent_b["Gene"])


def make_children(parent_a, parent_b, max_trq_set): 
    """ Make children """
    child1_gene = front_gene(parent_a["Gene"], max_trq_set) + back_gene(parent_b["Gene"], max_trq_set)
    child2_gene = front_gene(parent_b["Gene"], max_trq_set) + back_gene(parent_a["Gene"], max_trq_set)

    child1_chsm = {"Gene":gene_corrector(child1_gene), "Torque_list":TORQUE_LIST, 
                   "Presence_ratios":PRESENCE_RATIO, "Score":0.0}
    child2_chsm = {"Gene":gene_corrector(child2_gene), "Torque_list":TORQUE_LIST, 
                   "Presence_ratios":PRESENCE_RATIO, "Score":0.0}
    return child1_chsm, child2_chsm

def crossover(parent_a, parent_b):
    #if sum(parent_a["Torque_list"].values()) == 0.0: return
    #if sum(parent_b["Torque_list"].values()) == 0.0: return
    
    """ Extract sets """
    set1 = extract_sets(parent_a["Gene"])
    set2 = extract_sets(parent_b["Gene"])

    no_similar_sets = False
    similar_block_crossover_prob = random.randint(1,10)

    if similar_block_crossover_prob > 5: 
        """ Determine similar sets """
        similar_sets = similar_set_checker(set1, set2)
        if len(similar_sets) == 0: no_similar_sets = True
        if no_similar_sets == False: 
            
            """ Find highest torque in sets """
            avg_list = torque_average(parent_a["Torque_list"], parent_b["Torque_list"], 
                                                                similar_sets)
            max_trq_set = max_torque(avg_list, similar_sets)
            max_trq_set = max_trq_set[0] #["II"] --> II

            child1_chsm, child2_chsm = make_children(parent_a, parent_b, max_trq_set)  

    #Only crosses over highest set torques
    if similar_block_crossover_prob <= 5 or no_similar_sets: 
        parent_selector = random.randint(1,10)
        if parent_selector > 5: 
            #Sets parent_a as front half of child
            max_trq_set = max_set_torque(parent_a) 

        else: 
            #Sets parent_b as back half of child
            max_trq_set = max_set_torque(parent_b)

        
        child1_chsm, child2_chsm = make_children(parent_a, parent_b, max_trq_set)

    return child1_chsm, child2_chsm


#Adds one more block to model of block with highest presence ratio
def PRblock_buff(individual):
    best_block = max_model_block(individual)
    gene = individual["Gene"]
    best_block_in = False

    for idx in range(len(gene)-1,-1,-1):
        #Search for max presence ratio block
        if gene[idx][0] == best_block:
            pos = idx
            best_block_in = True
            break

    if best_block_in: gene.insert(pos+1, individual["Gene"][pos])

    mutant_chromosome = {"Gene":gene_corrector(gene), "Torque_list":TORQUE_LIST, 
                         "Presence_ratios":PRESENCE_RATIO, "Score":0.0}

    return mutant_chromosome

#Adds one more block to model of block with highest presence ratio
def add_block(individual):
    blocks = [['CrLu', 8, 8], ['Invr', 8, 8, 6, 2],
              ['Res',8,8],['Bot',8,2]]
    best_block = random.choice(blocks)
    gene = individual["Gene"]
    pos = random.randint(0,len(gene))

    gene.insert(pos,best_block)

    mutant_chromosome = {"Gene":gene_corrector(gene), "Torque_list":TORQUE_LIST, 
                         "Presence_ratios":PRESENCE_RATIO, "Score":0.0}

    return mutant_chromosome

#Removes most recent block with lowest presence ratio in model 
def removeBlock(individual):
    worst_block = min_model_block(individual) 
    gene = individual["Gene"]
    worst_block_in = False

    for idx in range(len(gene)-1,-1,-1):
        #Search for max presence ratio block
        if gene[idx][0] == worst_block:
            pos = idx
            worst_block_in = True
            break

    if worst_block_in: gene.remove(gene[pos])

    mutant_chromosome = {"Gene":gene_corrector(gene), "Torque_list":TORQUE_LIST, 
                         "Presence_ratios":PRESENCE_RATIO, "Score":0.0}

    return mutant_chromosome

#Removes random block in model   
def randomRemove(individual):
    block_length = len(individual["Gene"])
    remove_idx = random.randint(0,block_length-1)
    gene = individual["Gene"]
    gene.remove(gene[remove_idx])
    mutant_chromosome = {"Gene":gene_corrector(gene), "Torque_list":TORQUE_LIST, 
                         "Presence_ratios":PRESENCE_RATIO, "Score":0.0}

    return mutant_chromosome

def mutation(individual):
    options = [1,2,3,4]
    if sum(individual["Torque_list"].values()) == 0.0: return
    if sum(individual["Presence_ratios"].values()) == 0.0: return
    choice =  random.choice(options)
    #Add block with highest PR
    if choice == 1 or choice == 3: 
        mutant = PRblock_buff(individual)
        return mutant

    #Remove block with lowest PR
    if choice == 2: 
        mutant = removeBlock(individual)
        return mutant

    #Remove random block - 3
    if random.choice(options) == 3:
        mutant = randomRemove(individual)
        return mutant

    #Add random block - 4
    if random.choice(options) == 4:
        mutant = add_block(individual)
        return mutant


""" Initialize population of CNN models """
def initialize_pop(input_feature, size_of_pop):
    initial_population = []

    for ele in range(size_of_pop):
        initial_population.append(make_chromosome(input_feature))
    return initial_population

""" Rank fitness of all CNN models """ 
def rank_pop(population): 
    for ele in population: 
        Evo_model = Net(ele["Gene"])
        Train = Evaluate()
        score = Evaluate.fit(Evo_model)
        ele["Fitness"] = score
        #assign torque 
        #assign presence ratio


pop = [{'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Res', 8, 8], ['Res', 8, 8], ['Res', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 46.53},
       {'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['CrLu', 8, 8], ['CrLu', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 41.58},
       {'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['CrLu', 8, 8], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['CrLu', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 31.68},
       {'Gene': [['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['Res', 8, 8], ['Res', 8, 8], ['Res', 8, 8], ['Res', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 31.68},
       {'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Res', 8, 8], ['Res', 8, 8], ['Res', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 44.55},
       {'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['CrLu', 8, 8], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['CrLu', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 28.72},
       {'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 47.52},
       {'Gene': [['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['CrLu', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 28.72},
       {'Gene': [['Bot', 8, 2], ['CrLu', 8, 8], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Res', 8, 8], ['Res', 8, 8], ['Res', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 53.46},
       {'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Res', 8, 8], ['Res', 8, 8], ['Res', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 46.53}
       ]

# Update presence ratios
for model in pop: 
    invr = 0
    bot = 0 
    crlu = 0
    res = 0 
    block_length = len(model["Gene"])
    model_acc = model["Score"]
    # Counter for each block in gene
    for gene in model["Gene"]: 
        if gene[0] == 'Invr': invr += 1
        if gene[0] == 'Bot': bot += 1
        if gene[0] == 'CrLu': crlu += 1
        if gene[0] == 'Res': res += 1
    model['Presence_ratios']['Invr'] = model_acc * (invr/block_length)
    model['Presence_ratios']['Bot'] = model_acc * (bot/block_length)
    model['Presence_ratios']['CrLu'] = model_acc * (crlu/block_length)
    model['Presence_ratios']['Res'] = model_acc * (res/block_length)


# Update torque_list 
for model in pop: 
    model_acc = model["Score"]
    sets_in_model = extract_sets(model["Gene"])
    for sets in sets_in_model: 
        model["Torque_list"][sets[0]] = sets_in_model.count(sets) * model_acc


""" Breeding pool """
previous_top_performers = [{'Gene': [['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 0.0},
                           {'Gene': [['Bot', 8, 2], ['CrLu', 8, 8], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Res', 8, 8],['Res', 8, 8], ['Res', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 0.0},
                           {'Gene': [['Invr', 8, 8, 1, 1], ['Bot', 8, 2], ['CrLu', 2, 8], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['CrLu', 8, 8], ['Bot', 8, 2], ['CrLu', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 0.0},
                           {'Gene': [['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['CrLu', 8, 8], ['Invr', 8, 8, 6, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['Bot', 8, 2], ['CrLu', 8, 8], ['Res', 8, 8], ['Res', 8, 8]], 'Torque_list': {'BR': 1.0, 'BI': 1.0, 'BB': 1.0, 'BC': 1.0, 'RR': 1.0, 'RI': 1.0, 'RB': 1.0, 'RC': 1.0, 'CR': 1.0, 'CI': 1.0, 'CB': 1.0, 'CC': 1.0, 'IR': 1.0, 'II': 1.0, 'IB': 1.0, 'IC': 1.0}, 'Presence_ratios': {'Bot': 0.0, 'Res': 0.0, 'Invr': 0.0, 'Clru': 0.0}, 'Score': 0.0},]

for ele in previous_top_performers:
    pop.append(ele)
print (len(pop)) 

if __name__=="__main__":
    # Initialize population
    #pop = initialize_pop(8, 10)
    gen = 0
    end_gen = 10
    architectures = []


    while gen < end_gen: 

        offspring = []
        # Reproducing
        while len(offspring) < 10: 
            parent_a = random.choice(pop)
            parent_b = random.choice(pop)

            crossover_prob = random.randint(1,10)   
            if crossover_prob < 4:
                child1, child2 = crossover(parent_a, parent_b)
                offspring.append(child1)
                offspring.append(child2)
            else:
                mutant = mutation(random.choice(pop))
                offspring.append(mutant)

        #pop = pop + offspring
        # Check fitness
        #individual_gene = pop[0]["Gene"]
        #print (individual_gene)

        #model = Net(individual_gene)
        #test_result = Evaluate(model, 10)
        #Assign presence ratio and torque_list 
        #break
        # Rank 

        # Cull Lowest
        #pop = pop[:10]


        # Report architecture
        #architectures.append(pop)
        break
        gen += 1
print (offspring)


import random
from Modules import Net, Evaluate
gene = [['Invr', 32, 32, 1, 1], ['Invr', 32, 64, 6, 2],['Invr', 64, 96, 6, 2], ['Invr', 96, 160, 6, 2],
        ['Invr', 160, 320, 6, 1],['Invr', 320, 320, 6, 1],['Invr', 64, 64, 6, 1]]

gene = [['Invr', 32, 32, 1, 1], ['Invr', 32, 64, 6, 2],['Invr', 64, 96, 6, 2], ['Invr', 96, 160, 6, 2],
        ['Invr', 160, 320, 6, 1]]#,['Res', 320, 320],['Res', 320, 320]]
model = Net(gene)
test_result = Evaluate(model, num_epochs=15)

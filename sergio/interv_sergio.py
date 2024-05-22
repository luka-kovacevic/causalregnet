import numpy as np
from sergio.gene import *
from scipy.stats import ttest_rel, ttest_ind, ranksums
import sys
import csv
import networkx as nx
from scipy.stats import wasserstein_distance
import scipy.sparse as sp
import cdt

class Sergio(object):

    def __init__(self,
                 number_genes, 
                 number_bins, 
                 number_sc, 
                 noise_params = 1.0,
                 noise_type = "dpd",
                 decays = 0.8, 
                 kout = None,
                 kdown = None,
                 dynamics = False, 
                 sampling_state = 10, 
                 tol = 1e-3,
                 window_length = 100, 
                 dt = 0.01, 
                 optimize_sampling = False,
                 safety_steps = 0.0,
                 bifurcation_matrix = None, 
                 noise_params_splice = None, 
                 noise_type_splice = None,
                 splice_ratio = 4, 
                 dt_splice = 0.01, 
                 migration_rate = None):
        """
        Noise is a gaussian white noise process with zero mean and finite variance.
        noise_params: The amplitude of noise in CLE. This can be a scalar to use
        for all genes or an array with the same size as number_genes.
        Tol: p-Value threshold above which convergence is reached
        window_length: length of non-overlapping window (# time-steps) that is used to realize convergence
        dt: time step used in  CLE
        noise_params and decays: Could be an array of length number_genes, or single value to use the same value for all genes
        number_sc: number of single cells for which expression is simulated
        sampling_state (>=1): single cells are sampled from sampling_state * number_sc steady-state steps
        optimize_sampling: useful for very large graphs. If set True, may help finding a more optimal sampling_state and so may ignore the input sampling_state
        noise_type: We consider three types of noise, 'sp': a single intrinsic noise is associated to production process, 'spd': a single intrinsic noise is associated to both
        production and decay processes, 'dpd': two independent intrinsic noises are associated to production and decay processes
        dynamics: whether simulate splicing or not
        bifurcation_matrix: is a numpy array (nBins_ * nBins) of <1 values; bifurcation_matrix[i,j] indicates whether cell type i differentiates to type j or not. Its value indicates the rate of transition. If dynamics == True, this matrix should be specified
        noise_params_splice: Same as "noise_params" but for splicing. if not specified, the same noise params as pre-mRNA is used
        noise_type_splice: Same as "noise_type" but for splicing. if not specified, the same noise type as pre-mRNA is used
        splice_ratio: it shows the relative amount of spliced mRNA to pre-mRNA (at steady-state) and therefore tunes the decay rate of spliced mRNA as a function of unspliced mRNA. Could be an array of length number_genes, or single value to use the same value for all genes
        dt_splice = time step for integrating splice SDE


        Note1: It's assumed that no two or more bins differentiate into the same new bin i.e. every bin has either 0 or 1 parent bin
        Note2: differentitation rates (e.g. type1 -> type2) specified in bifurcation_matrix specifies the percentage of cells of type2 that are at the vicinity of type1
        """

        self.nGenes_ = number_genes
        self.nBins_ = number_bins
        self.nSC_ = number_sc
        self.sampling_state_ = sampling_state
        self.tol_ = tol
        self.winLen_ = window_length
        self.dt_ = dt
        self.optimize_sampling_ = optimize_sampling
        self.safety_steps = safety_steps
        self.level2verts_ = {}
        self.gID_to_level_and_idx = {} # This dictionary gives the level and idx in self.level2verts_ of a given gene ID
        self.binDict = {} # This maps bin ID to list of gene objects in that bin; only used for dynamics simulations
        self.maxLevels_ = 0
        self.init_concs_ = np.zeros((number_genes, number_bins))
        self.meanExpression = -1 * np.ones((number_genes, number_bins))
        self.noiseType_ = noise_type
        self.dyn_ = dynamics
        self.nConvSteps = np.zeros(number_bins) # This holds the number of simulated steps till convergence
        ## dynamics capabilities removed 
        # if dynamics:
        #     self.bifurcationMat_ = np.array(bifurcation_matrix)
        #     self.binOrders_ = []
        #     self.binDict = {}
        #     for b in range(self.nBins_):
        #         self.binDict[b] = np.zeros(self.nGenes_,).tolist()
        ############
        # This graph stores for each vertex: parameters(interaction
        # parameters for non-master regulators and production rates for master
        # regulators), tragets, regulators and level
        ############
        self.graph_ = {}

        if kout is None:
            kout = np.zeros(number_genes).astype(bool)
        self.kout = kout

        if kdown is None:
            kdown = np.ones(number_genes)
        self.kdown = kdown

        if np.isscalar(noise_params):
            self.noiseParamsVector_ = np.repeat(noise_params, number_genes)
        elif np.shape(noise_params)[0] == number_genes:
            self.noiseParamsVector_ = noise_params
        else:
            print ("Error: expect one noise parameter per gene")


        if np.isscalar(decays) == 1:
            self.decayVector_ = np.repeat(decays, number_genes)
        elif np.shape(decays)[0] == number_genes:
            self.decayVector_ = decays
        else:
            print ("Error: expect one decay parameter per gene")
            sys.exit()

        ## dynamics capabilities removed 
        # if self.dyn_:
        #     if (self.bifurcationMat_ == None).any():
        #         print ("Error: Bifurcation Matrix is missing")
        #         sys.exit()

        #     if noise_type_splice == None:
        #         self.noiseTypeSp_ = noise_type
        #     else:
        #         self.noiseTypeSp_ = noise_type_splice


        #     if dt_splice == None:
        #         self.dtSp_ = np.copy(self.dt_)
        #     else:
        #         self.dtSp_ = dt_splice


        #     if noise_params_splice == None:
        #         self.noiseParamsVectorSp_ = np.copy(self.noiseParamsVector_)
        #     elif np.isscalar(noise_params_splice):
        #         self.noiseParamsVectorSp_ = np.repeat(noise_params_splice, number_genes)
        #     elif np.shape(noise_params_splice)[0] == number_genes:
        #         self.noiseParamsVectorSp_ = noise_params_splice
        #     else:
        #         print ("Error: expect one splicing noise parameter per gene")
        #         sys.exit()

        #     if np.isscalar(splice_ratio):
        #         self.ratioSp_ = np.repeat(splice_ratio, number_genes)
        #     elif np.shape(splice_ratio)[0] == number_genes:
        #         self.ratioSp_ = splice_ratio
        #     else:
        #         print ("Error: expect one splicing ratio parameter per gene")
        #         sys.exit()

    """
    Relies on networkx package to infer master regulators from DiGraph structure. 
    """
    def get_master_regs(self, dag):
        mr_list = [n for n,d in dag.in_degree() if d == 0]
        return(mr_list)

    def generate_random_dag(self, max_parents=None, 
                              min_k_param=1.0, max_k_param=5.0, 
                              min_b_param=1.0, max_b_param=3.0, # basal production rate for MR should depend on data set
                              min_hill_coeff=2.0, max_hill_coeff=2.0,
                              generate_params=True, graph=None, dag_type='erdos', expected_degree=None):

        """ Generates DAG and corresponding parameters for SERGIO. If DAG is provided, only parameters are 
        generated.

        Args:
            max_parents: maximum number of parents for any node
            min_x_param: lower bound for uniform distribution from which 'x' parameter is drawn
            max_x_param: upper bound for uniform distribution from which 'x' parameter is drawn
            generate_params: are SERGIO parameters generate or only DAG
            graph: If None a random DAG is used; otherwise expecting networkx object
            dag_type: Method used for DAG generation
            expected_degree: Number of edges expected 
        """

        # generate graph

        assert type(generate_params) is bool
        assert min_k_param <= max_k_param or generate_params == False
        assert min_b_param <= max_b_param or generate_params == False
        assert min_hill_coeff <= max_hill_coeff or generate_params == False

        if dag_type == 'erdos':
            assert expected_degree != None
        elif dag_type == 'default':
            assert max_parents != None

        if graph == None:

            dag_generator = cdt.data.AcyclicGraphGenerator("linear", 
                                                           nodes = self.nGenes_, 
                                                           parents_max = max_parents, 
                                                           npoints = 1, 
                                                           dag_type=dag_type, 
                                                           expected_degree=expected_degree)
            graph = dag_generator.generate()[1]
            graph = nx.convert_node_labels_to_integers(graph)

        self.nx_graph = graph

        assert nx.is_directed_acyclic_graph(graph), 'Generated graph is not a directed acyclic graph. Try again or set a different seed.'
        assert nx.number_of_selfloops(graph) == 0, 'Generated graph has self-loops. Try again or set a different seed.'

        mr_list = self.get_master_regs(graph)
        self.master_regulators_idx_ = set(mr_list)

        graph_mat = nx.adjacency_matrix(graph)
 
        if generate_params:

            k_params_mat = graph_mat.copy()
            k_params_mat.data = np.random.uniform(low = min_k_param, high = max_k_param, size = nx.number_of_edges(graph))

            hill_coeffs_mat = graph_mat.copy()
            hill_coeffs_mat.data =  np.random.uniform(low = min_hill_coeff, high = max_hill_coeff, size = nx.number_of_edges(graph))

            mr_row = np.repeat(mr_list, self.nBins_)
            mr_row = np.concatenate(([0], mr_row))

            mr_col = np.concatenate(np.repeat([range(0, self.nBins_)], len(mr_list), axis = 0))
            mr_col = np.concatenate(([0], mr_col))

            b_params = np.random.uniform(low = min_b_param, high = max_b_param, size = len(mr_list)*self.nBins_)
            b_params = np.concatenate(([0], b_params))
            assert len(b_params) == len(mr_col) == len(mr_row)
            b_params_mat = sp.csr_matrix((b_params, (mr_row, mr_col)), shape=(self.nGenes_, self.nBins_))

            return graph_mat, k_params_mat, b_params_mat, hill_coeffs_mat

        return graph_mat

    def custom_graph(self, *, g, k, b, hill):
        """
        Prepare custom graph model and coefficients
        Args:
            g: [nGenes_, nGenes_] graph
            k: [nGenes_, nGenes_] interaction coeffs
            b: [nGenes_, nBins_] basal reproduction rate of source nodes (master regulators)
            hill: [nGenes_, nGenes_] hill coefficients of nonlinear interactions
        """

        # check inputs
        assert g.shape == k.shape
        assert g.shape == hill.shape
        assert g.shape[0] == self.nGenes_ # temporary removal
        assert g.shape[1] == self.nGenes_
        assert b.shape[0] == self.nGenes_
        assert b.shape[1] == self.nBins_
        assert np.allclose(g[np.diag_indices(g.shape[0])], 0.0), f"No self loops allowed"

        # following steps of original function
        for i in range(self.nGenes_):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []


        self.master_regulators_idx_ = set()

        for j in range(self.nGenes_):

            is_parent = g[:, j]

            # master regulator (no parents)
            if is_parent.sum() == 0:

                self.master_regulators_idx_.add(j)
                self.graph_[j]['rates'] = b[j]
                self.graph_[j]['regs'] = []
                self.graph_[j]['level'] = -1

            # regular gene (target)
            else:

                currInteraction = []
                currParents = []
                for u in np.where(is_parent == 1)[0]:
                    currInteraction.append((u, k[u, j], hill[u, j], 0))  # last zero shows half-response, it is modified in another method
                    currParents.append(u)
                    self.graph_[u]['targets'].append(j)

                self.graph_[j]['params'] = currInteraction
                self.graph_[j]['regs'] = currParents
                self.graph_[j]['level'] = -1  # will be modified later


        self.find_levels_(self.graph_)

    """
    Method allows for reading sparse adjacency matrices into SERGIO. Wider compatibility than simply networkx. Can be used with other graph generating packages. 
    """
    def build_adj_graph(self,
                       g,
                       k_params,
                       b_params,
                       hill_coeffs):

        assert g.shape == k_params.shape
        assert g.shape == hill_coeffs.shape
        assert b_params.shape[0] == g.shape[0]
        assert b_params.shape[1] == self.nBins_
        # assert g.shape[0] == self.nGenes_ # temporary removal only
        # assert g.shape[1] == self.nGenes_
        
        assert sp.issparse(g)
        assert sp.issparse(k_params)
        assert sp.issparse(b_params)
        assert sp.issparse(hill_coeffs)

        for gID in range(self.nGenes_):
            self.graph_[gID] = {}
            self.graph_[gID]['targets'] = []

        mr_list = []

        for gID in range(self.nGenes_):
            currInteraction = []
            currParents = []

            if gID in g.nonzero()[1]:
                # target

                # iterate through regulators
                for regID in g.getcol(gID).nonzero()[0]:

                    currInteraction.append((np.int(regID), np.float(k_params[regID, gID]), np.float(hill_coeffs[regID, gID]), 0))
                    currParents.append(np.int(regID))
                    self.graph_[np.int(regID)]['targets'].append(gID)

                self.graph_[np.int(gID)]['params'] = currInteraction
                self.graph_[np.int(gID)]['regs'] = currParents
                self.graph_[np.int(gID)]['level'] = -1

            else: 
                
                self.graph_[np.int(gID)]['rates'] = b_params.getrow(gID)
                self.graph_[np.int(gID)]['regs'] = []
                self.graph_[np.int(gID)]['level'] = -1
                mr_list.append(gID)

        self.master_regulators_idx_ = set(mr_list)
        self.find_levels_(self.graph_)


    """
    Method for integrating networkx DiGraphs with SERGIO simulator. Facilitates reading of networkx DiGraph to format for SERGIO.
    """
    def build_nx_graph(self,
                       dag): # networkx DiGraph object where each edge has k_param and hill_coeff attributes -- master regulators have b_param attributes
        
        # assert dag.shape == k_params.shape
        # assert dag.shape == hill_coeffs.shape
        # assert b_params.shape[0] == dag.shape[0]
        # assert b_params.shape[1] == self.nBins_
        # assert dag.shape[0] == self.nGenes_
        # assert dag.shape[1] == self.nGenes_

        allRegs = []
        allTargets = []

        self.master_regulators_idx_ = set(self.get_master_regs(dag = dag))

        for gID in range(self.nGenes_):
            self.graph_[gID] = {}
            self.graph_[gID]['targets'] = []
            
            currInteraction = []
            currParents = []

            if any(True for _ in dag.predecessors(gID)):

                for regID in dag.predecessors(gID):

                    currInteraction.append(np.int(regID), np.float(dag.nodes[regID]['k_params']), np.float(dag.nodes[regID]['hill_coeff']), 0)
                    currParents.append(np.int(regID))
                    self.graph_[np.int(regID)]['targets'].append(gID)

                self.graph_[np.int(gID)]['params'] = currInteraction
                self.graph_[np.int(gID)]['regs'] = currParents
                self.graph_[np.int(gID)]['level'] = -1

            else: 

                self.graph_[np.int(gID)]['rates'] = dag.nodes[gID]['b_params']
                self.graph_[np.int(gID)]['regs'] = []
                self.graph_[np.int(gID)]['level'] = -1

        self.find_levels_(self.graph_)

    def build_graph (self, input_file_taregts, input_file_regs, shared_coop_state = 0):
        """
        # 1- shared_coop_state: if >0 then all interactions are modeled with that
        # coop state, and coop_states in input_file_taregts are ignored. Otherwise,
        # coop states are read from input file. Reasonbale values ~ 1-3
        # 2- input_file_taregts: a csv file, one row per targets. Columns: Target Idx, #regulators,
        # regIdx1,...,regIdx(#regs), K1,...,K(#regs), coop_state1,...,
        # coop_state(#regs)
        # 3- input_file_regs: a csv file, one row per master regulators. Columns: Master regulator Idx,
        # production_rate1,...,productions_rate(#bins)
        # 4- input_file_taregts should not contain any line for master regulators
        # 5- For now, assume that nodes in graph are either master regulator or
        # target. In other words, there should not be any node with no incomming
        # or outgoing edge! OTHERWISE IT CAUSES ERROR IN CODE.
        # 6- The indexing of genes start from 0. Also, the indexing used in
        # input files should match the indexing (if applicable) used for initilizing
        # the object.
        """

        for i in range(self.nGenes_):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []


        allRegs = []
        allTargets = []

        with open(input_file_taregts,'r') as f:
            reader = csv.reader(f, delimiter=',')
            if (shared_coop_state <= 0):
                for row in reader:
                    nRegs = np.int(row[1])
                    ##################### Raise Error ##########################
                    if nRegs == 0:
                        print ("Error: a master regulator (#Regs = 0) appeared in input")
                        sys.exit()
                        ############################################################

                    currInteraction = []
                    currParents = []
                    for regId, K, C_state in zip(row[2: 2 + nRegs], row[2+nRegs : 2+2*nRegs], row[2+2*nRegs : 2+3*nRegs]):
                        currInteraction.append((np.int(regId), np.float(K), np.float(C_state), 0)) # last zero shows half-response, it is modified in another method
                        allRegs.append(np.int(regId))
                        currParents.append(np.int(regId))
                        self.graph_[np.int(regId)]['targets'].append(np.int(row[0]))

                    self.graph_[np.int(row[0])]['params'] = currInteraction
                    self.graph_[np.int(row[0])]['regs'] = currParents
                    self.graph_[np.int(row[0])]['level'] = -1 # will be modified later
                    allTargets.append(np.int(row[0]))

                    #if self.dyn_:
                    #    for b in range(self.nBins_):
                    #        binDict[b].append(gene(np.int(row[0]),'T', b))
            else:
                for indRow, row in enumerate(reader):
                    nRegs = np.int(np.float(row[1]))
                    ##################### Raise Error ##########################
                    if nRegs == 0:
                        print ("Error: a master regulator (#Regs = 0) appeared in input")
                        sys.exit()
                        ############################################################

                    currInteraction = []
                    currParents = []
                    for regId, K, in zip(row[2: 2 + nRegs], row[2+nRegs : 2+2*nRegs]):
                        currInteraction.append((np.int(np.float(regId)), np.float(K), shared_coop_state, 0)) # last zero shows half-response, it is modified in another method
                        allRegs.append(np.int(np.float(regId)))
                        currParents.append(np.int(np.float(regId)))
                        self.graph_[np.int(np.float(regId))]['targets'].append(np.int(np.float(row[0])))

                    self.graph_[np.int(np.float(row[0]))]['params'] = currInteraction
                    self.graph_[np.int(np.float(row[0]))]['regs'] = currParents
                    self.graph_[np.int(np.float(row[0]))]['level'] = -1 # will be modified later
                    allTargets.append(np.int(np.float(row[0])))

                    #if self.dyn_:
                    #    for b in range(self.nBins_):
                    #        binDict[b].append(gene(np.int(row[0]),'T', b))

        #self.master_regulators_idx_ = set(np.setdiff1d(allRegs, allTargets))

        with open(input_file_regs,'r') as f:
            masterRegs = []
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if np.shape(row)[0] != self.nBins_ + 1:
                    print ("Error: Inconsistent number of bins")
                    sys.exit()

                masterRegs.append(int(float(row[0])))
                self.graph_[int(float(row[0]))]['rates'] = [np.float(i) for i in row[1:]]
                self.graph_[int(float(row[0]))]['regs'] = []
                self.graph_[int(float(row[0]))]['level'] = -1

                #if self.dyn_:
                #    for b in range(self.nBins_):
                #        binDict[b].append(gene(np.int(row[0]),'MR', b))

        self.master_regulators_idx_ = set(masterRegs)


        if (len(self.master_regulators_idx_) + np.shape(allTargets)[0] != self.nGenes_):
            print ("Error: Inconsistent number of genes")
            sys.exit()

        self.find_levels_(self.graph_) # make sure that this modifies the graph

        if self.dyn_:
            self.find_bin_order_(self.bifurcationMat_)

    def find_levels_ (self, graph):
        """
        # This is a helper function that takes a graph and assigns layer to all
        # verticies. It uses longest path layering algorithm from
        # Hierarchical Graph Drawing by Healy and Nikolovself. A bottom-up
        # approach is implemented to optimize simulator run-time. Layer zero is
        # the last layer for which expression are simulated
        # U: verticies with an assigned layer
        # Z: vertizies assigned to a layer below the current layer
        # V: set of all verticies (genes)

        This also sets a dictionary that maps a level to a matrix (in form of python list)
        of all genes in that level versus all bins

        Note to self:
        This is like DFS topsort, but compressing the length of levels as much as possible
        Essentially, root nodes have the highest level (to be simulated first) and sink nodes have level 0,
        and any node upstream of a node has higher level
        Sets:
            level2verts_
                {l: [M, bins] where M is number of genes in on level l}

            gID_to_level_and_idx
                {v: (level, j) where j in index of vertex v in `level2verts[level]` for vertex v in graph
        """

        U = set()
        Z = set()
        V = set(graph.keys())

        currLayer = 0
        self.level2verts_[currLayer] = []
        idx = 0

        while U != V:
            currVerts = set(filter(lambda v: set(graph[v]['targets']).issubset(Z), V-U))

            for v in currVerts:
                graph[v]['level'] = currLayer
                U.add(v)
                if {v}.issubset(self.master_regulators_idx_):
                    allBinList = [gene(v,'MR', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1
                else:
                    allBinList = [gene(v,'T', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1

            currLayer += 1
            Z = Z.union(U)
            self.level2verts_[currLayer] = []
            idx = 0

        self.level2verts_.pop(currLayer)
        self.maxLevels_ = currLayer - 1

        if not self.dyn_:
            self.set_scIndices_()

    def set_scIndices_ (self):
        """
        # First updates sampling_state_ if optimize_sampling_ is set True: to optimize run time,
        run for less than 30,000 steps in first level
        # Set the single cell indices that are sampled from steady-state steps
        # Note that sampling should be performed from the end of Concentration list
        # Note that this method should run after building graph(and layering) and should
        be run just once!
        """

        if self.optimize_sampling_:
            state = np.true_divide(30000 - self.safety_steps * self.maxLevels_, self.nSC_)
            if state < self.sampling_state_:
                self.sampling_state_ = state

        # time indices when to collect "single-cell" expression snapshots
        self.scIndices_ = np.random.randint(low = - self.sampling_state_ * self.nSC_, high = 0, size = self.nSC_)

    def calculate_required_steps_(self, level):
        """
        # Calculates the number of required simulation steps after convergence at each level.
        # safety_steps: estimated number of steps required to reach convergence (same), although it is not neede!
        """
        #TODO: remove this safety step

        # Note to self: as safety measure leaving this safety step to double check
        # that knockouts/knockdowns have reached steady-state
        # however, we do initialize the concentrations correctly so should be fine without it

        return self.sampling_state_ * self.nSC_ + level * self.safety_steps

    def calculate_half_response_(self, level):
        """
        Calculates the half response for all interactions between previous layer
        and current layer
        """

        currGenes = self.level2verts_[level]

        for g in currGenes: # g is list of all bins for a single gene
            c = 0
            if g[0].Type == 'T':
                for interTuple in self.graph_[g[0].ID]['params']:
                    regIdx = interTuple[0]
                    meanArr = self.meanExpression[regIdx]

                    if set(meanArr) == set([-1]):
                        print ("Error: Something's wrong in either layering or simulation. Expression of one or more genes in previous layer was not modeled.")
                        sys.exit()

                    self.graph_[g[0].ID]['params'][c] = (self.graph_[g[0].ID]['params'][c][0], self.graph_[g[0].ID]['params'][c][1], self.graph_[g[0].ID]['params'][c][2], np.mean(meanArr))
                    c += 1
            #Else: g is a master regulator and does not need half response

    def hill_(self, reg_conc, half_response, coop_state, repressive = False):
        """
        So far, hill function was used in the code to model 1 interaction at a time.
        So the inputs are single values instead of list or array. Also, it models repression based on this assumption.
        if decided to make it work on multiple interaction, repression should be taken care as well.
        """
        if reg_conc == 0:
            if repressive:
                return 1
            else:
                return 0
        else:
            if repressive:
                return 1 - np.true_divide(np.power(reg_conc, coop_state), (np.power(half_response, coop_state) + np.power(reg_conc, coop_state)) )
            else:
                return np.true_divide(np.power(reg_conc, coop_state), (np.power(half_response, coop_state) + np.power(reg_conc, coop_state)) )

    def init_gene_bin_conc_ (self, level):
        """
        Initilizes the concentration of all genes in the input level

        Note: calculate_half_response_ should be run before this method
        """

        currGenes = self.level2verts_[level]
        for g in currGenes:
            gID = g[0].ID

            interv_factor = self.kdown[gID]

            # # adjust initial concentratino for possible knockout/knockdown experiment
            # if self.kout[gID]:
            #     assert not self.kdown[gID], "Cannot knockout and knockdown the same gene"
            #     print('knocking down:'+str(gID))
            #     interv_factor = 0.0
            # elif self.kdown[gID]:
            #     interv_factor = 0.1
            # else:
            #     interv_factor = 1.0

            # initialize at expected steady-state
            if g[0].Type == 'MR':
                allBinRates = self.graph_[gID]['rates']

                for bIdx, rate in enumerate(allBinRates):
                   g[bIdx].append_Conc(np.true_divide(interv_factor * rate, self.decayVector_[g[0].ID]))

            else:
                params = self.graph_[g[0].ID]['params']

                for bIdx in range(self.nBins_):
                    rate = 0
                    for interTuple in params:
                        meanExp = self.meanExpression[interTuple[0], bIdx]
                        rate += np.abs(interTuple[1]) * self.hill_(meanExp, interTuple[3], interTuple[2], interTuple[1] < 0)

                    g[bIdx].append_Conc(np.true_divide(interv_factor * rate, self.decayVector_[g[0].ID]))

    def calculate_prod_rate_(self, bin_list, level):
        """
        calculates production rates for the input list of gene objects in different bins but all associated to a single gene ID
        """
        type = bin_list[0].Type

        if (type == 'MR'):
            rates = self.graph_[bin_list[0].ID]['rates']
            return np.array([rates[gb.binID] for gb in bin_list])

        else:
            params = self.graph_[bin_list[0].ID]['params']
            Ks = [np.abs(t[1]) for t in params]
            regIndices = [t[0] for t in params]
            binIndices = [gb.binID for gb in bin_list]
            currStep = bin_list[0].simulatedSteps_
            lastLayerGenes = np.copy(self.level2verts_[level + 1])
            hillMatrix = np.zeros((len(regIndices), len(binIndices)))

            for tupleIdx, rIdx in enumerate(regIndices):
                regGeneLevel = self.gID_to_level_and_idx[rIdx][0]
                regGeneIdx = self.gID_to_level_and_idx[rIdx][1]
                regGene_allBins = self.level2verts_[regGeneLevel][regGeneIdx]
                for colIdx, bIdx in enumerate(binIndices):
                    hillMatrix[tupleIdx, colIdx] = self.hill_(regGene_allBins[bIdx].Conc[currStep], params[tupleIdx][3], params[tupleIdx][2], params[tupleIdx][1] < 0)

            return np.matmul(Ks, hillMatrix)


    def CLE_simulator_(self, level):

        self.calculate_half_response_(level)
        self.init_gene_bin_conc_(level)
        nReqSteps = self.calculate_required_steps_(level)

        # list of lists of genes at the current level
        # each inner list contains `nBins` elements, representing expressions of genes of a given cell type (bin) at time t
        # level2verts_: {level: [[...], ...., [...]]}
        sim_set = np.copy(self.level2verts_[level]).tolist()

        # simulate expression of each gene at this level (vectorized for all bins)
        while sim_set != []:

            delIndicesGenes = []

            # g: list of gene objects of length 'nBins'
            for gi, bin_list in enumerate(sim_set):

                # gene id (row/col in adjacency matrix)
                gID = bin_list[0].ID

                # level in graph, index in list of expressions per bin (same as gi)
                gLevel, gIDX = self.gID_to_level_and_idx[gID]
                assert level == gLevel, "Levels should match"
                assert gi == gIDX, "index in gene-bin matrix should match"

                # [nBins,] current expressions
                currExp = np.array([gb.Conc[-1] for gb in bin_list])

                # if knocked out, set production rate to 0
                # if knocked down, multiply production rate by set
                if self.kout[gID]:
                    prod_rate = np.zeros(len(currExp))
                elif self.kdown[gID] != 1:
                    prod_rate = self.kdown[gID] * self.calculate_prod_rate_(bin_list, level)
                else:
                    prod_rate = self.calculate_prod_rate_(bin_list, level)

                # [nBins,] decay rate
                decay = np.multiply(self.decayVector_[gID], currExp)

                # sample noise
                if self.noiseType_ == 'sp':
                    # This notation is inconsistent with our formulation, dw should
                    # include dt^0.5 as well, but here we multipy dt^0.5 later
                    # [nBins, ]
                    dw = np.random.normal(size = len(currExp))
                    amplitude = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5))
                    noise = np.multiply(amplitude, dw)

                elif self.noiseType_ == "spd":
                    # [nBins, ]
                    dw = np.random.normal(size = len(currExp))
                    amplitude = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5) + np.power(decay, 0.5))
                    noise = np.multiply(amplitude, dw)


                elif self.noiseType_ == "dpd":
                    # [nBins, ]
                    dw_p = np.random.normal(size = len(currExp))
                    dw_d = np.random.normal(size = len(currExp))

                    amplitude_p = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5))
                    amplitude_d = np.multiply (self.noiseParamsVector_[gID] , np.power(decay, 0.5))

                    noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)

                else:
                    raise KeyError(f"Unknown noise type {self.noiseType_}")

                # [nBins,] change in expression per bin
                dxdt = self.dt_ * (prod_rate - decay) + np.power(self.dt_, 0.5) * noise

                # update expression for each bin
                delIndices = []
                for bIDX, gObj in enumerate(bin_list):

                    # append new concentration level to list of expressions in bin
                    binID = gObj.binID
                    gObj.append_Conc(gObj.Conc[-1] + dxdt[bIDX])
                    gObj.incrementStep()

                    # check whether we collected enough samples
                    if len(gObj.Conc) == nReqSteps:
                        # if so, extract and save expressions at preset time snapshots
                        gObj.set_scExpression(self.scIndices_)
                        self.meanExpression [gID, binID] = np.mean(gObj.scExpression)
                        self.level2verts_[level][gIDX][binID] = gObj
                        delIndices.append(bIDX)

                # remove bins to be simulated when they are done
                sim_set[gi] = [i for j, i in enumerate(bin_list) if j not in delIndices]

                if sim_set[gi] == []:
                    delIndicesGenes.append(gi)

            # remove genes to be simulated if done
            sim_set = [i for j, i in enumerate(sim_set) if j not in delIndicesGenes]


    def simulate(self):
        for level in range(self.maxLevels_, -1, -1):
            self.CLE_simulator_(level)


    def getExpressions(self):
        ret = np.zeros((self.nBins_, self.nGenes_, self.nSC_))
        for l in range(self.maxLevels_ + 1):
            currGeneBins = self.level2verts_[l]
            for g in currGeneBins:
                gIdx = g[0].ID

                for gb in g:
                    ret[gb.binID, gIdx, :] = gb.scExpression

        return ret

    """""""""""""""""""""""""""""""""""""""
    "" This part is to add technical noise
    """""""""""""""""""""""""""""""""""""""
    def outlier_effect(self, scData, outlier_prob, mean, scale):
        """
        Args:
            scData: shape [#cell_types, #genes, #cells_per_type].
        """
        out_indicator = np.random.binomial(n = 1, p = outlier_prob, size = self.nGenes_)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = np.random.lognormal(mean = mean, sigma = scale, size = numOutliers)
        ##################################

        scData = np.concatenate(scData, axis = 1)
        for i, gIndx in enumerate(outlierGenesIndx):
            scData[gIndx,:] = scData[gIndx,:] * outFactors[i]

        # return np.split(scData, self.nBins_, axis = 1) # BUG in original code; should return same shape as input
        return np.stack(np.split(scData, self.nBins_, axis = 1))


    def lib_size_effect(self, scData, mean, scale):
        """
        This functions adjusts the mRNA levels in each cell seperately to mimic
        the library size effect. To adjust mRNA levels, cell-specific factors are sampled
        from a log-normal distribution with given mean and scale.

        scData: the simulated data representing mRNA levels (concentrations);
        np.array (#bins * #genes * #cells)

        mean: mean for log-normal distribution

        var: var for log-normal distribution

        returns libFactors ( np.array(nBin, nCell) )
        returns modified single cell data ( np.array(nBin, nGene, nCell) )
        """

        #TODO make sure that having bins does not intefere with this implementation
        ret_data = []

        libFactors = np.random.lognormal(mean = mean, sigma = scale, size = (self.nBins_, self.nSC_))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = np.sum(binExprMatrix, axis = 0 )
            binFactors = binFactors / np.where(normalizFactors == 0.0, 1.0, normalizFactors)
            binFactors = binFactors.reshape(1, self.nSC_)
            binFactors = np.repeat(binFactors, self.nGenes_, axis = 0)

            ret_data.append(np.multiply(binExprMatrix, binFactors))


        return libFactors, np.array(ret_data)


    def dropout_indicator(self, scData, shape = 1, percentile = 65):
        """
        This is similar to Splat package

        Input:
        scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)

        shape: the shape of the logistic function

        percentile: the mid-point of logistic functions is set to the given percentile
        of the input scData

        returns: np.array containing binary indactors showing dropouts
        """
        scData = np.array(scData)
        scData_log = np.log(np.add(scData,1))
        log_mid_point = np.percentile(scData_log, percentile)
        prob_ber = np.true_divide (1, 1 + np.exp( -1*shape * (scData_log - log_mid_point) ))

        binary_ind = self.random.binomial( n = 1, p = prob_ber)

        return binary_ind

    def convert_to_UMIcounts (self, scData):
        """
        Input: scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)
        """
        return np.random.poisson(scData)
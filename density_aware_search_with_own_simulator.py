'''
Authored by Sihyung Lee
'''

import logging
import math
import os
import random


def simulate_quantum_counting(N, M, m):
    '''
    Simulate the estimation of solution count by Quantum Counting
    
    Input:
        N: size of search space
        M: true number of solutions
        m: the number of counting qubits
    
    Output:
        (1) M_estimated: an estimate of M, obtained with Quantum Counting
        (2) total number of oracle calls invoked by Quantum Counting
    '''
    numMeasuredValues = 2**m
    minError, M_estimated_with_minimum_error = None, None
    for measuredValue in range(0, numMeasuredValues):
        theta = 2 * math.pi * (measuredValue / numMeasuredValues)
        numTargetsEstimated = N * (math.sin(theta/2)**2)
        if minError is None or abs(numTargetsEstimated - M) < minError:
            minError, M_estimated_with_minimum_error = abs(numTargetsEstimated - M), numTargetsEstimated

    M_estimated_with_minimum_error = int(M_estimated_with_minimum_error + 0.5) # round the estimated value to the nearest integer
    M_estimated_with_minimum_error = min(int(math.sqrt(N)), M_estimated_with_minimum_error) # we assume that M <= sqrt(N)

    num_oracle_calls = 2**m - 1
    logging.debug(f"With N({N}), M({M}), and {m} counting qubits, quantum counting estimated {M_estimated_with_minimum_error} with {num_oracle_calls} oracle invocations")    

    return M_estimated_with_minimum_error, num_oracle_calls


def generate_answers_in_one_subspace(N, M, S):
    '''
    Generate solutions within a single subspace 111...111xxx...xxx, where some elements are solutions and others are not
        The solutions are randomly selected from the subspace

    Input:
        N: size of search space
        M: number of solutions
        S: size of subspace (S >= M)

    Output:
        (1) answerMap: dictionary of answers with (key, value) = (answer, a Solution instance corresponding to the answer)
        (2) answerSetBinary: answerSetBinary[i][j] contains the set of answers with i-th bit == j,
            where i = 0 ~ log2(N) - 1 and j = 0 or 1 
    '''
    assert math.log2(N) == int(math.log2(N))    
    assert math.log2(S) == int(math.log2(S))
    assert S >= M

    all_elements_in_subspace = [i for i in range(N - 1, N - S - 1, -1)]
    random.shuffle(all_elements_in_subspace)
    answerMap = {}
    for i in range(M): answerMap[all_elements_in_subspace[i]] = Solution(all_elements_in_subspace[i], N)        

    result = ['1' for _ in range(int(math.log2(N) - math.log2(S)))]
    for _ in range(int(math.log2(S))): result.append('x')
    logging.info(f"generate one subspace of the form {''.join(result)}, where {M} solutions and {S - M} non-solutions are randomly mixed")
    logging.debug(f"the selected solutions are: {answerMap.values()}")

    return answerMap, create_binary_answer_set(N, answerMap)


def generate_answers_in_two_subspaces(N, M, S):
    '''
    Generate solutions within two equal-sized subspaces 111...111xxx...xxx and 000...000xxx...xxx, 
        where some elements in each space might not be solutions, depending on S

    Input:
        N: size of search space
        M: number of solutions
        S: Size of subspace (S >= M // 2)

    Output:
        (1) answerMap: dictionary of answers with (key, value) = (answer, a Solution instance corresponding to the answer)
        (2) answerSetBinary: answerSetBinary[i][j] contains the set of answers with i-th bit == j,
            where i = 0 ~ log2(N) - 1 and j = 0 or 1 
    '''
    assert math.log2(N) == int(math.log2(N))
    assert math.log2(M) == int(math.log2(M))
    num_solutions_in_one_space = M // 2
    assert S >= num_solutions_in_one_space

    answerMap = {}
    
    # cluster 1: 111...111xxx...xxx
    all_elements_in_subspace = [i for i in range(N - 1, N - S - 1, -1)]
    random.shuffle(all_elements_in_subspace)    
    for i in range(num_solutions_in_one_space): answerMap[all_elements_in_subspace[i]] = Solution(all_elements_in_subspace[i], N)        

    # cluster 2: 000...000xxx...xxx      
    all_elements_in_subspace = [i for i in range(S)]
    random.shuffle(all_elements_in_subspace)
    for i in range(num_solutions_in_one_space): answerMap[all_elements_in_subspace[i]] = Solution(all_elements_in_subspace[i], N)        

    result1 = ['1' for _ in range(int(math.log2(N - S)))]
    result2 = ['0' for _ in range(int(math.log2(N - S)))]
    for _ in range(int(math.log2(S))): 
        result1.append('x')
        result2.append('x')
    logging.info(f"generate two subspaces of the form {''.join(result1)} and {''.join(result2)} with each size {S} and solutions {num_solutions_in_one_space}")
    logging.debug(f"the selected solutions are: {answerMap.values()}")

    return answerMap, create_binary_answer_set(N, answerMap)


def generate_answers_in_four_subspaces(N, M, S):
    '''
    Generate solutions within four equal-sized subspaces 111...111xxx...xxx, 000...000xxx...xxx, 
        000...111xxx...xxx (equal numbers of 0's and 1's exist in the prefix), and
        111...000xxx...xxx (equal numbers of 0's and 1's exist in the prefix)
        where some elements in each space might not be solutions, depending on S

    Input:
        N: size of search space
        M: number of solutions
        S: Size of subspace (S >= M // 4)

    Output:
        (1) answerMap: dictionary of answers with (key, value) = (answer, a Solution instance corresponding to the answer)
        (2) answerSetBinary: answerSetBinary[i][j] contains the set of answers with i-th bit == j,
            where i = 0 ~ log2(N) - 1 and j = 0 or 1 
    '''
    assert math.log2(N) == int(math.log2(N))
    assert math.log2(M) == int(math.log2(M))
    num_solutions_in_one_space = M // 4
    assert S >= num_solutions_in_one_space

    answerMap = {}
    
    # cluster 1: 111...111xxx...xxx
    all_elements_in_subspace = [i for i in range(N - 1, N - S - 1, -1)]
    random.shuffle(all_elements_in_subspace)    
    for i in range(num_solutions_in_one_space): 
        answerMap[all_elements_in_subspace[i]] = Solution(all_elements_in_subspace[i], N)
        #logging.info(answerMap[all_elements_in_subspace[i]])    

    # cluster 2: 000...000xxx...xxx      
    all_elements_in_subspace = [i for i in range(S)]
    random.shuffle(all_elements_in_subspace)
    for i in range(num_solutions_in_one_space): 
        answerMap[all_elements_in_subspace[i]] = Solution(all_elements_in_subspace[i], N)
        #logging.info(answerMap[all_elements_in_subspace[i]])    

    # cluster 3: 000...111xxx...xxx
    startingNumber = int(N ** 0.75 - N ** 0.5)
    all_elements_in_subspace = [startingNumber + i for i in range(S)]
    random.shuffle(all_elements_in_subspace)
    for i in range(num_solutions_in_one_space): 
        answerMap[all_elements_in_subspace[i]] = Solution(all_elements_in_subspace[i], N)
        #logging.info(answerMap[all_elements_in_subspace[i]])    

    # cluster 4: 111...000xxx...xxx
    startingNumber = int(N - N**0.75)
    all_elements_in_subspace = [startingNumber + i for i in range(S)]
    random.shuffle(all_elements_in_subspace)
    for i in range(num_solutions_in_one_space): 
        answerMap[all_elements_in_subspace[i]] = Solution(all_elements_in_subspace[i], N)
        #logging.info(answerMap[all_elements_in_subspace[i]])    

    logging.info(f"generate four subspaces with each size {S} and solutions {num_solutions_in_one_space}")
    logging.debug(f"the selected solutions are: {answerMap.values()}")

    return answerMap, create_binary_answer_set(N, answerMap)


def generate_answers_randomly(N, M):
    '''
    Generate solutions uniformly randomly within the entire search space

    Input:
        N: size of search space
        M: number of solutions

    Output:
        (1) answerMap: dictionary of answers with (key, value) = (answer, a Solution instance corresponding to the answer)
        (2) answerSetBinary: answerSetBinary[i][j] contains the set of answers with i-th bit == j,
            where i = 0 ~ log2(N) - 1 and j = 0 or 1 
    '''
    assert math.log2(N) == int(math.log2(N))    

    answerMap = {}
    for _ in range(M):
        r = random.randint(0, N - 1)
        while r in answerMap: r = random.randint(0, N - 1)
        answerMap[r] = Solution(r, N)   
    
    logging.info(f"generate {M} solutions uniformly randomly from the entire search space with {N} elements")
    logging.debug(f"the selected solutions are: {answerMap.values()}")

    return answerMap, create_binary_answer_set(N, answerMap)


def create_binary_answer_set(N, answerMap):
    '''
    Create an answer set in the binary format

    Input: 
        N: size of search space
        answerMap: dictionary of answers with (key, value) = (answer, a Solution instance corresponding to the answer)

    Output: answerSetBinary, where answerSetBinary[i][j] contains the set of answers with i-th bit == j,
            i = 0 ~ log2(N) - 1 and j = 0 or 1 
    '''
    assert math.log2(N) == int(math.log2(N))
    n = math.ceil(math.log2(N))
    maxValue = 2**n - 1

    answerSetBinary = [[set(), set()] for _ in range(n)]
    for value, solution in answerMap.items():
        assert value <= maxValue, f"answer {value} must be <= {maxValue}"        
        for i in range(n):
            answerSetBinary[i][solution.valueInBinaryList[i]].add((value, solution))

    return answerSetBinary


MAX_FAIL_STREAK = 10
def simulate_density_aware_search(N, M_estimated, answerMap, answerSetBinary):
    '''
    Simulate the discovery of all solutions using Grover's algorithm,
        while dynamically adjusting the search scope based on observed solution density
    
    Input:

    Output:
        (1) total number of solutions discovered by the baseline method
        (2) total number of oracle calls made by the baseline method 
    '''
    assert math.log2(N) == int(math.log2(N))
    n = math.ceil(math.log2(N))

    #
    # Configure thresholds
    #    
    maxNumNones = n // 2   # a subspace cannot grow larger than one that includes more than sqrt(N) elements, since we assume M <= sqrt(N)
    minSpaceSizeToFocus = math.sqrt(M_estimated)  # we focus on a subspace only when it grows larger than this threshold, since focusing on too small subspace does not help improve efficiency
    maxNumSubspaces = math.log2(N) // 2 # we limit the total number of subspaces under this threshold, so that the computational complexity of maintaining subspaces remains moderate
    
    fail_streak = 0

    search_space_global = SearchSpace(0, None)
    search_space_global.initialize_as_global_search_space(N, n, answerMap, M_estimated) 
    search_space = search_space_global
    search_space_map = {0: search_space_global} # hashmap that stores all subspaces. key: space id, value: SearchSpace instance

    num_discovered_solutions = 0
    total_num_oracle_calls = 0
    
    while True:
        #
        # Simulate one repetition of Grover's algorithm
        #
        Grover_iterations = math.floor(math.pi / 4 * math.sqrt(search_space.size() / search_space.M_estimated))
        total_num_oracle_calls += (Grover_iterations + 1)  # '+1' for verifyting whether the measured element is a solution
        probability_measuring_solutions = math.sin((2 * Grover_iterations + 1) * math.asin(math.sqrt(search_space.solution_count() / search_space.size()))) ** 2

        logging.debug(f"M_estimated = {search_space.M_estimated} with {Grover_iterations + 1}/{total_num_oracle_calls} oracle calls and probability {probability_measuring_solutions} of measuring solutions")        

        if random.random() < probability_measuring_solutions:  # random.random() returns a real number x, where 0.0 <= x < 1.0            
            #
            # Case 1: Grover's algorithm discovered a solution (i.e., a success occurred)
            #
            fail_streak = 0
            num_discovered_solutions += 1

            #
            # Randomly choose one element in the current search scope to simualte a randomly sampled solution            
            #
            solution = search_space.answerMap[random.choice(list(search_space.answerMap.keys()))]
            logging.debug(f"Grover's algorithm found a solution: {solution}")

            #
            # Remove the sampled solution s from all subspaces that contains s, to simulate the exclusion of s from the oracle            
            #
            spaceWithMaxCount, maxCount = None, None
            for space in search_space_map.values():
                if solution.value in space.answerMap:                    
                    space.answer_found(solution)
                    logging.debug(f"remove solution {solution} from space {space}")
                    if space is not search_space_global and spaceWithMaxCount is None:
                        spaceWithMaxCount, maxCount = space, n

            if search_space is search_space_global:
                logging.debug(f"current search space is global: {search_space}")
                # 
                # if the current search space is global,
                #   then include the found solution into the best-matching subspace S, while updating its pattern
                #  
                #   if S qualifies as a new focus:
                #       if S's oracle simplifies into TRUE, then discover all elements in S as solutions
                #       otherwise, if S's density > global density, move focus to S
                #
                
                #
                # if the found solution s does not belong to any of the subspaces, 
                #   find one that matches the most bits and update its pattern to include s
                #
                if spaceWithMaxCount is None:
                    logging.debug(f"solution {solution} does not belong to any subspaces, so find the best match")                
                    for id, space in search_space_map.items():
                        if id == 0: continue  # skip the global space
                        matchCount = space.count_match(solution)                        
                        if (n - matchCount) + space.numNones > maxNumNones: continue # if, including s into this subspace grows it larger than the maximum allowed size, then skip it. (thus we also skip the global search space)
                        if maxCount is None or matchCount > maxCount:
                            spaceWithMaxCount, maxCount = space, matchCount
                    if spaceWithMaxCount is not None:                        
                        spaceWithMaxCount.include_answer(solution, answerSetBinary)
                        logging.debug(f"inject solution {solution} into the best matching space: {spaceWithMaxCount}")
                
                if spaceWithMaxCount is not None:
                    #
                    # check to see if the subspace that just included s qualifies as a focus
                    #
                    if len(spaceWithMaxCount.answersFound) >= math.log2(spaceWithMaxCount.numElements) and\
                    spaceWithMaxCount.size() >= minSpaceSizeToFocus:                        
                        assert spaceWithMaxCount.size() >= len(spaceWithMaxCount.answerMap) + len(spaceWithMaxCount.answersFound),\
                            f"subspace {spaceWithMaxCount} has {spaceWithMaxCount.size()} elements, which must be >= the number of solutions in the subspace {len(spaceWithMaxCount.answerMap)} unfound + {len(spaceWithMaxCount.answersFound)} found"
                        logging.debug(f"subspace {spaceWithMaxCount} qualifies as a new focus")
                        if spaceWithMaxCount.size() == len(spaceWithMaxCount.answerMap) + len(spaceWithMaxCount.answersFound):
                            #
                            # if, the oracle for the subspace simplifies into true (i.e., all elements in the subspace are solutions)
                            #   then discover all solutions in the subspace and remove the subspace from the subspace map
                            #
                            logging.debug(f"The subspace's oracle simplies into true, and thus discover all its elements as solutions and delete the subspace")
                            allSolutionsInSpace = list(spaceWithMaxCount.answerMap.values())
                            for solution in allSolutionsInSpace:
                                num_discovered_solutions += 1
                                for space in search_space_map.values():
                                    if solution.value in space.answerMap:
                                        space.answer_found(solution)
                            del search_space_map[spaceWithMaxCount.id]                        
                        else:
                            #
                            # Otherwise, check to see if the subspace has a larger density than the global space
                            #                                                      
                            error_margin = math.sqrt(spaceWithMaxCount.size())
                            m = math.floor(math.log2(spaceWithMaxCount.size() / error_margin))                            
                            M_estimated_new, num_oracle_calls_by_quantum_counting = simulate_quantum_counting(spaceWithMaxCount.size(), len(spaceWithMaxCount.answerMap), m)
                            total_num_oracle_calls += num_oracle_calls_by_quantum_counting
                            logging.debug(f"estimate the number of solutions in the subspace: {M_estimated_new}/{len(spaceWithMaxCount.answerMap)} with {num_oracle_calls_by_quantum_counting}/{total_num_oracle_calls} oracle calls")
                            M_estimated_new = len(spaceWithMaxCount.answerMap) # Assume that the estimate is correct    
                            spaceWithMaxCount.M_estimated = M_estimated_new

                            density_current_space = search_space.M_estimated / search_space.size()
                            density_new_space = spaceWithMaxCount.M_estimated / spaceWithMaxCount.size()
                            logging.debug(f"the density of current space {density_current_space:.5f} and new space {density_new_space:.5f}")
                            if density_new_space > density_current_space:
                                search_space = spaceWithMaxCount
                                logging.debug(f"start focusing on subspace {search_space}, as its density > global density")                                
                            else:
                                del search_space_map[spaceWithMaxCount.id]
                                logging.debug(f"remove the subspace, as its density <= global density")
                    else:
                        #
                        # if the subspace does not yet qualify as a focus, then proceed to collect more solutions
                        #
                        pass
                else:
                    #
                    # if no subspace exists that can include the found soution s, create a new subspace that includes s
                    #                    
                    search_space_new = SearchSpace(len(search_space_map), solution)
                    search_space_map[len(search_space_map)] = search_space_new # hashmap that stores all subspaces. key: space id, value: SearchSpace instance    
                    logging.debug(f"no subspace include the solution, and thus create a new subspace {search_space_new}")

                    if len(search_space_map) > maxNumSubspaces:
                        #
                        # if the number of subspaces exceed a threshold, remove one with the least priority, to maintain a reasonable level of computation time
                        #
                        spaceWithMinPriority = None
                        for space in search_space_map.values():
                            if space.id == len(search_space_map) - 1: continue  # do not remove the newly-created subspace
                            if spaceWithMinPriority is None or space < spaceWithMinPriority:
                                spaceWithMinPriority = space                        
                        logging.debug(f"more than {maxNumSubspaces} and thus remove the least-priority subspace {spaceWithMinPriority}")
                        del search_space_map[spaceWithMinPriority.id]

            else:                
                #
                # if the current search space is not global, 
                #   its pattern does not need to change, as the found solution belongs to this space
                #
                logging.debug(f"current search space is NOT global: {search_space}")                     

        else:
            #
            # Case 2: Grover's algorithm failed to discover a solution
            #            
            fail_streak += 1            
            logging.debug(f"Grover's algorithm failed to find a solution {fail_streak} consecutive times")
            if fail_streak > MAX_FAIL_STREAK: # If no solution is collected for MAX_FAIL_STREAK consecutive attempts, terminate the repetitions within this searc space, assuming that no more solutions remain
                #
                # if the current search scope is the global scope, then break, since no solution can be found globally
                # otherwise, reset the search scope as the global scope, since no solution can be found locally within the current scope
                #
                logging.debug(f"the fail streak {fail_streak} > max threshold {MAX_FAIL_STREAK}")
                if search_space is search_space_global:
                    logging.debug(f"terminate since the search space is global")
                    break
                else:
                    logging.debug(f"remove the current search space {search_space} and fall back to the global search space")                    
                    del search_space_map[search_space.id]
                    search_space = search_space_global
                    fail_streak = 0

    return num_discovered_solutions, total_num_oracle_calls


class Solution:
    def __init__(self, valueInInteger, N):
        assert math.log2(N) == int(math.log2(N))    
        n = math.ceil(math.log2(N))
        self.value = valueInInteger
        self.valueInBinaryString = format(valueInInteger, f'0{n}b')
        self.valueInBinaryList = [int(self.valueInBinaryString[i]) for i in range(n)]
        self.found = False

    def __str__(self):
        return f"{self.valueInBinaryString}({self.value})"
    
    def __repr__(self):
        return self.__str__()


class SearchSpace:
    def __init__(self, id, solution):
        self.id = id
        self.numNones = 0 # number of Nones in the pattern
        self.numElements = 1     # total number of elements within this search space

        if solution is not None:
            self.pattern = solution.valueInBinaryList.copy()
            self.answersFound = {solution.value:solution} # solutions already found
        else:    
            self.pattern = [] # each element is 0, 1, or None, where None means that both 0 and 1 are possible (i.e., a "don't care" bit)
            self.answersFound = {}  # solutions already found

        self.answerMap = {} # remaining solutions not yet found                
        self.M_estimated = 1 # estimated number of solution 

    def initialize_as_global_search_space(self, N, n, answerMap, M_estimated):
        self.pattern = [None] * n
        self.numNones = n
        self.numElements = N
        self.answerMap = dict(answerMap)  # creates a shallow copy of answerMap (i.e., a new independent dictionary with the same elements)        
        self.M_estimated = M_estimated

    def count_match(self, answer):
        matchCount = 0
        for i in range(len(self.pattern)):
            if self.pattern[i] is None or answer.valueInBinaryList[i] == self.pattern[i]: matchCount += 1
        return matchCount

    def include_answer(self, answer, answerSetBinary):
        #
        # include an answer that does not belong to this subspace, by modifying its pattern to include more answers
        #
        answerSet = set()
        for i in range(len(self.pattern)):
            if self.pattern[i] is not None and self.pattern[i] != answer.valueInBinaryList[i]:
                self.pattern[i] = None
                self.numNones += 1
                self.numElements *= 2
            if self.pattern[i] is None:
                if i == 0:
                    answerSet.update(answerSetBinary[i][0])
                    answerSet.update(answerSetBinary[i][1])
                else:
                    pass  # the answerSet remains as it is and no need to reduce it by intersection
            else:
                if i == 0: answerSet.update(answerSetBinary[i][self.pattern[i]])
                else: answerSet.intersection_update(answerSetBinary[i][self.pattern[i]])

        self.answersFound[answer.value] = answer
        for answer, solution in answerSet:
            if not solution.found and answer not in self.answerMap: self.answerMap[answer] = solution
            elif solution.found and answer not in self.answersFound: self.answersFound[answer] = solution
            #if answer not in self.answerMap and answer not in self.answersFound: self.answerMap[answer] = solution               

    def solution_count(self): # return the number of solutions within this space
        return len(self.answerMap)
    
    def size(self):
        return self.numElements
    
    def answer_found(self, answer): 
        #
        # if an answer in answerMap is found, move it to answersFound
        #
        if answer.value in self.answerMap: 
            solution = self.answerMap.pop(answer.value)
            solution.found = True
            self.M_estimated = max(1, self.M_estimated - 1) # M_estimated must be >= 1, so that the resulting number of Grover's iterations remains > 0
            self.answersFound[answer.value] = solution

    def __str__(self):
        #
        # return a string representation of this subspace's pattern
        #
        result = []
        for e in self.pattern:
            if e == 0 or e == 1: result.append(str(e))
            else: result.append('x')  # 'x' represents a "don't care" bit
        result = [''.join(result)]
        result.append("with solutions found {")
        for s in self.answersFound.values(): result.append(str(s))            
        result.append("} and unfound {")
        for s in self.answerMap.values(): result.append(str(s))
        result.append("}")
        return ' '.join(result)

    def __repr__(self):
        return self.__str__()
    
    def __lt__(self, other):
        if len(self.answersFound) != len(other.answersFound): return len(self.answersFound) < len(other.answersFound)  # a subspace with more solutions found is more important
        else: return self.id < other.id  # a subspace more recently created is more important


def simulate_search_with_baseline_method(N, M, M_estimated):
    '''
    Simulate the discovery of all solutions using Grover's algorithm 
        without adjusting the search scope (as in previous studies)

    Input:
        N: size of search space
        M: true number of solutions
        M_estimated: an estimated of M        

    Output:
        (1) total number of solutions discovered by the baseline method
        (2) total number of oracle calls made by the baseline method    
    '''
    num_discovered_solutions = 0
    total_num_oracle_calls = 0
    fail_streak = 0

    while True:
        #
        # Simulate one repetition of Grover's algorithm
        #
        Grover_iterations = math.floor(math.pi / 4 * math.sqrt(N / M_estimated))
        total_num_oracle_calls += (Grover_iterations + 1)  # '+1' for confirming the collected solution
        probability_measuring_solutions = math.sin((2 * Grover_iterations + 1) * math.asin(math.sqrt(M / N))) ** 2

        logging.debug(f"M_estimated = {M_estimated} with {Grover_iterations + 1}/{total_num_oracle_calls} oracle calls and probability {probability_measuring_solutions} of measuring solutions")

        if random.random() < probability_measuring_solutions:  # random.random() returns a real number x, where 0.0 <= x < 1.0            
            #
            # Grover's algorithm discovered a solution (i.e., a success occurred)
            #
            num_discovered_solutions += 1
            M_estimated = max(1, M_estimated - 1)
            M = M - 1
            fail_streak = 0
        else:
            #
            # Grover's algorithm failed to discover a solution
            #
            fail_streak += 1            
            if fail_streak > MAX_FAIL_STREAK: break  # If no solution is collected for MAX_FAIL_STREAK consecutive attempts, terminate the repetitions, assuming that M_estimated > M

    return num_discovered_solutions, total_num_oracle_calls


if __name__ == "__main__":
    '''
    Create a log file and configure the logging level
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current Python script
    log_file_path = os.path.join(script_dir, "simulation_density_aware_search.log")  # Ensure the log file is saved in the same directory as this Python script
    logging.basicConfig(
        filename = log_file_path,
        level = logging.INFO,  # logging levels: debug < info < warning < error < critical        
        format = "%(asctime)s - %(levelname)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    #
    # Unit test for simulate_density_aware_search() in comparison with simulate_search_with_baseline_method()
    #        
    for rho in [0.125, 0.25, 0.5]:
        
        for N in [2**14, 2**16, 2**18, 2**20]:
            num_simulations = 10
            collection_num_discovered_solutions = []
            collection_total_num_oracle_calls = []
            
            M = int(math.sqrt(N))
            error_margin = math.sqrt(N)
            m = math.floor(math.log2(N / error_margin)) # number of counting qubits
            logging.info(f"N = {N}, M = {M}, rho = {rho:.3f}, 1 subspace")
            
            #
            # Uncomment only one of the following four lines
            #
            answerMap, answerSetBinary = generate_answers_in_one_subspace(N, M, int(M / rho))
            #answerMap, answerSetBinary = generate_answers_randomly(N, M)
            #answerMap, answerSetBinary = generate_answers_in_two_subspaces(N, M, int(M / rho / 2))
            #answerMap, answerSetBinary = generate_answers_in_four_subspaces(N, M, int(M / rho / 4))

            M_estimated, num_oracle_calls_by_quantum_counting = simulate_quantum_counting(N, M, m)

            #
            # Unit test for simulate_search_with_baseline_method()
            #
            num_discovered_solutions, total_num_oracle_calls = simulate_search_with_baseline_method(N, M, M_estimated)
            logging.info(f"Baseline method: with N = {N}, M = {M}, and M_estimated = {M_estimated}, {num_discovered_solutions} solutions are discovered with {num_oracle_calls_by_quantum_counting} + {total_num_oracle_calls} oracle calls")
            
            #
            # Unit test for simulate_density_aware_search()
            #
            for _ in range(num_simulations):   
                num_discovered_solutions, total_num_oracle_calls = simulate_density_aware_search(N, M_estimated, answerMap, answerSetBinary)        
                collection_num_discovered_solutions.append(num_discovered_solutions)
                collection_total_num_oracle_calls.append(total_num_oracle_calls)
                for solution in answerMap.values():
                    solution.found = False

            average_num_discovered_solutions = sum(collection_num_discovered_solutions) / len(collection_num_discovered_solutions)
            average_total_num_oracle_calls = sum(collection_total_num_oracle_calls) / len(collection_total_num_oracle_calls)
            logging.info(f"Density-aware search: on average, {average_num_discovered_solutions:.2f}/{M} solutions are discoverd with M_estimated = {M_estimated} and {num_oracle_calls_by_quantum_counting} + {average_total_num_oracle_calls:.2f} oracle calls")
            
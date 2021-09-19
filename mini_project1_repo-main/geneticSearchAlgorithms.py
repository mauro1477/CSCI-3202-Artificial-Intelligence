from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
from random import choices 
import math 
import random 
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams
from matplotlib import pyplot as plt 

class GASolver: 
    def __init__(self, params, lst_of_identifiers, n):
        # Parameters for GA: see geneticAlgParams
        # Also includes test data for regression and checking validity
        self.params = params
        # The population size 
        self.N = n
        # Store the actual population (you can use other data structures if you wish)
        self.pop = []
        # A list of identifiers for the expressions
        self.identifiers = lst_of_identifiers
        # Maintain statistics on best fitness in each generation
        self.population_stats = []
        # Store best solution so far across all generations
        self.best_solution_so_far = None
        # Store the best fitness so far across all generations
        self.best_fitness_so_far = -float('inf')

    # Please add whatever helper functions you wish.

    # TODO: Implement the genetic algorithm as described in the
    # project instructions.
    # This function need not return anything. However, it should
    # update the fields best_solution_so_far, best_fitness_so_far and
    # population_stats
    def run_ga_iterations(self, n_iter=1000):
        i = 0
        while i < self.N:
            expr = generate_random_expr(self.params.depth, self.identifiers, self.params)
            if is_viable_expr(expr, self.identifiers, self.params):
                self.pop.append(expr)
                i = i + 1
                
        previous_best_solution_so_far = self.best_solution_so_far
        previous_best_fitness_so_far  = self.best_fitness_so_far
        
        for main_index in range(0, n_iter):   
#             print(main_index)
            current_population = []
            for expr in self.pop:
                score = compute_fitness(expr, self.identifiers, self.params)
                current_population.append((expr,score))

            sorted_population = []
            sorted_population = sorted(current_population, key=lambda tup: tup[1], reverse=True)
            k = self.params.elitism_fraction * self.N

            elitism = []
            for i in range(0, int(k)):
                elitism.append(sorted_population[i])

            lst_expr = []
            weights = []
            for y in current_population:
                lst_expr.append(y[0])
                new_weight = math.exp(y[1]/self.params.temperature)
                weights.append(new_weight)


            weights = tuple(weights)
            result = []

            i = 0
            while i < self.N - k:
                hello = random.choices(lst_expr, weights=weights, k=2)
                e1 = hello[0]
                e2 = hello[1]

                (e1_random_subtree, e2_random_subtree) = random_subtree_crossover(e1, e2, True)

                e1_cross = random_expression_mutation(e1_random_subtree, self.identifiers, self.params, True)
                e2_cross = random_expression_mutation(e2_random_subtree, self.identifiers, self.params, True)

                if is_viable_expr(e1_cross,self.identifiers, self.params):
                    result.append(e1_cross)
                    i+=1

                if is_viable_expr(e2_cross,self.identifiers, self.params):
                    result.append(e2_cross)               
                    i+=1

            children = []
            for child in result:
                score = compute_fitness(child, self.identifiers, self.params)
                children.append((child,score))

            next_generation = elitism + children  

            sorted_next_generation = []
            sorted_next_generation = sorted(next_generation, key=lambda tup: tup[1], reverse=True)                
            
#             print("sort next gen", sorted_next_generation)
            best_expr_next = sorted_next_generation[0][0]
            best_score_next = sorted_next_generation[0][1]
            
            if best_score_next > previous_best_fitness_so_far:
                self.best_fitness_so_far = best_score_next
                self.best_solution_so_far = best_expr_next
                self.population_stats.append(self.best_fitness_so_far)
                
                previous_best_fitness_so_far = best_score_next
                previous_best_solution_so_far = best_expr_next
            else:
                self.population_stats.append(previous_best_fitness_so_far)
                self.best_fitness_so_far = previous_best_fitness_so_far
                self.best_solution_so_far = previous_best_solution_so_far
        
        
            
        

## Function: curve_fit_using_genetic_algorithms
# Run curvefitting using given parameters and return best result, best fitness and population statistics.
# DO NOT MODIFY
def curve_fit_using_genetic_algorithm(params, lst_of_identifiers, pop_size, num_iters):
    solver = GASolver(params, lst_of_identifiers, pop_size)
    solver.run_ga_iterations(num_iters)
    return (solver.best_solution_so_far, solver.best_fitness_so_far, solver.population_stats)


# Run test on a toy problem.
if __name__ == '__main__':
    params = GAParams()
    params.regression_training_data = [
       ([-2.0 + 0.02*j], 5.0 * math.cos(-2.0 + 0.02*j) - math.sin((-2.0 + 0.02*j)/10.0)) for j in range(201)
    ]
    params.test_points = list([ [-4.0 + 0.02 * j] for j in range(401)])
#     solver = GASolver(params,['x'],500)
#     solver.run_ga_iterations(20)
    (sol, fit, pop) = curve_fit_using_genetic_algorithm(params, ['x'], 500, 100)
    print('Done!')
    print(f'Best solution found: {sol.simplify()}, fitness = {fit}')
    stats = pop
    niters = len(stats)
    plt.plot(range(len(stats)), [st for st in stats], 'b-')
    plt.show()

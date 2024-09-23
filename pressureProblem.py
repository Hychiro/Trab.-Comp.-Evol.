import numpy as np
from mealpy import PSO, FloatVar
from mealpy.utils.problem import Problem
import numpy as np
import mealpy as mp

GA = mp.evolutionary_based.GA
CSA = mp.swarm_based.CSA
flag = True
penalidade = [100000, 1000000, 10, 1]



def verifica(solution1, solution2):
    def g1(z):
        return -z[0] + 0.0193 * z[2]
    
    def g2(z):
        return -z[1] + 0.00954 * z[2]
    
    def g3(z):
        return  (- np.pi * z[2]**2 * z[3])  +(-(4/3) * np.pi * z[2]**3) + 1296000.0
    
    def g4(z):
        return z[3] - 240.0

    # Função para calcular penalidade
    def violate(value):
        return 0 if value <= 0 else value
    
    flag1 = False
    if violate(g1(solution1)) != 0 or violate(g1(solution2)) != 0:
        penalidade[0] = penalidade[0]*10
        flag1 = True
    if violate(g2(solution1)) != 0 or violate(g2(solution2)) != 0:
        penalidade[1] = penalidade[1]*10
        flag1 = True
    if violate(g3(solution1)) != 0 or violate(g3(solution2)) != 0:
        penalidade[2] = penalidade[2]*10
        flag1 = True
    if violate(g4(solution1)) != 0 or violate(g4(solution2)) != 0:
        penalidade[3] = penalidade[3]*10
        flag1 = True
    
    return flag1
    


while flag:
    def objective_function(solution):
        # Definindo as restrições
        def g1(z):
            return -z[0] + 0.0193 * z[2]
        
        def g2(z):
            return -z[1] + 0.00954 * z[2]
        
        def g3(z):
            return  (- np.pi * z[2]**2 * z[3])  +(-(4/3) * np.pi * z[2]**3) + 1296000.0
        
        def g4(z):
            return z[3] - 240.0

        # Função para calcular penalidade
        def violate(value):
            return 0 if value <= 0 else value

        # Função objetivo
        z1, z2, z3, z4 = solution
        p1,p2,p3,p4 = penalidade
        fx = 0.6224 * z1 * z3 * z4 + 1.7781 * z2 * z3**2 + 3.1661 * z1**2 * z4 + 19.84 * z1**2 * z3
        # Penalidades para violações de restrições
        fx = violate(g1(solution))*p1 + violate(g2(solution))*p2 + violate(g3(solution))*p3 + violate(g4(solution))*p4 + fx
        return fx

    # Dicionário do problema com função objetivo, limites e tipo de otimização
    problem_constrained = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[0, 0, 10., 10.], ub=[99., 99., 200., 200.]),  # Definir os limites adequados
        "minmax": "min",
    }
    # term = {
    # "max_early_stop": 250
    # }

    tournamentTestCaseResult = []
    tournamentTestCaseFitness= []
    tournamentTestCaseModels= []

    csaCaseResult = []
    csaCaseFitness = []
    csaCaseModels = []
    name = "pressureVessel"


    # Definir o modelo e resolver o problema
    for _ in range(30):
        model = GA.EliteMultiGA(epoch=1000, pop_size=50, pc=0.9, k_way=0.5, selection="tournament", crossover="multi_points")
        result = model.solve(problem_constrained )
        tournamentTestCaseResult.append(result.solution)
        tournamentTestCaseFitness.append(result.target.fitness)
        tournamentTestCaseModels.append(model)

    for _ in range(30):
        model = CSA.OriginalCSA(epoch=1000, pop_size=50,p_a=0.7)
        result = model.solve(problem_constrained)
        csaCaseResult.append(result.solution)
        csaCaseFitness.append(result.target.fitness)
        csaCaseModels.append(model)


    tournamentTestCaseResult = np.array(tournamentTestCaseResult)
    tournamentTestCaseFitness = np.array(tournamentTestCaseFitness)
    tournamentTestCaseModels = np.array(tournamentTestCaseModels)

    csaCaseResult = np.array(csaCaseResult)
    csaCaseFitness = np.array(csaCaseFitness)
    csaCaseModels =np.array(csaCaseModels)

    f = open(f"{name}/CasoPressureWithPenalityFINAL.txt", "a")
    f.write(f"============== === =============="+"\n")
    f.write(f"penalizacoes: {penalidade}")
    f.write("GA com torneio e multipontos:"+"\n")
    f.write("Media fitness de 10 iteracoes: " + str(np.sum(tournamentTestCaseFitness) / 30) +"\n")
    f.write("Melhor fitness de 10 iteracoes: " + str(np.min(tournamentTestCaseFitness))+"\n")
    val = tournamentTestCaseResult[np.argmin(tournamentTestCaseFitness)]
    f.write("Ponto de melhor fitness de 10 iteracoes: [" + str(val[0])+","+str(val[1])+","+str(val[2])+","+str(val[3])+"]\n")

    f.write("CSA:"+"\n")
    f.write("Media fitness de 10 iteracoes: " + str(np.sum(csaCaseFitness) / 30)+"\n")
    f.write("Melhor fitness de 10 iteracoes: " + str(np.min(csaCaseFitness))+"\n")
    val = csaCaseResult[np.argmin(csaCaseFitness)]
    f.write("Ponto de melhor fitness de 10 iteracoes: [" + str(val[0])+","+str(val[1])+","+str(val[2])+","+str(val[3])+"]\n")
    f.close()
    flagAux1 = verifica(csaCaseResult[np.argmin(csaCaseFitness)],tournamentTestCaseResult[np.argmin(tournamentTestCaseFitness)])
    print(flagAux1)
    model1 = csaCaseModels[np.argmin(csaCaseFitness)]    
    model1.history.save_global_objectives_chart(filename=f"{name}/csaCase/goc")
    model1.history.save_local_objectives_chart(filename=f"{name}/csaCase/loc")
    model1.history.save_global_best_fitness_chart(filename=f"{name}/csaCase/gbfc")
    model1.history.save_local_best_fitness_chart(filename=f"{name}/csaCase/lbfc")
    model1.history.save_runtime_chart(filename=f"{name}/csaCase/rtc")
    model1.history.save_exploration_exploitation_chart(filename=f"{name}/csaCase/eec")

    
    model2 = tournamentTestCaseModels[np.argmin(tournamentTestCaseFitness)]
    model2.history.save_global_objectives_chart(filename=f"{name}/tournamentWithMultipoints/goc")
    model2.history.save_local_objectives_chart(filename=f"{name}/tournamentWithMultipoints/loc")
    model2.history.save_global_best_fitness_chart(filename=f"{name}/tournamentWithMultipoints/gbfc")
    model2.history.save_local_best_fitness_chart(filename=f"{name}/tournamentWithMultipoints/lbfc")
    model2.history.save_runtime_chart(filename=f"{name}/tournamentWithMultipoints/rtc")
    model2.history.save_exploration_exploitation_chart(filename=f"{name}/tournamentWithMultipoints/eec")


    if not flagAux1:
        flag = False


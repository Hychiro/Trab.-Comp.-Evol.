# main.py

import numpy as np
from constants import FUNC_5, FUNC_9,PROBLEM_DICT_5,PROBLEM_DICT_9,NAME_5,NAME_9,FUNC_CONVEX, FUNC_NCONVEX,NAME_CONVEX,NAME_NCONVEX, PROBLEM_DICT_CONVEX10, PROBLEM_DICT_NCONVEX10, PROBLEM_DICT_CONVEX, PROBLEM_DICT_NCONVEX, TERM_DICT
from cases import TrabCases
from plotting import plot_heatmap_with_points, plot_heatmap_with_points_testCase,plot_heatmap_with_points_CaseWithCSA

# Create and run test cases for convex and non-convex functions
def run_test_cases():
    cases10dim1 = TrabCases()
    for _ in range(10):
        cases10dim1.randomWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX10)
        cases10dim1.tournamentWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX10)
        cases10dim1.randomWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX10)
        cases10dim1.tournamentWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX10)
    cases10dim1.npArray()

    cases10dim2 = TrabCases()
    for _ in range(10):
        cases10dim2.randomWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX10)
        cases10dim2.tournamentWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX10)
        cases10dim2.randomWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX10)
        cases10dim2.tournamentWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX10)
    cases10dim2.npArray()

    cases2dim1 = TrabCases()
    for _ in range(10):
        cases2dim1.randomWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX)
        cases2dim1.tournamentWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX)
        cases2dim1.randomWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX)
        cases2dim1.tournamentWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX)
    cases2dim1.npArray()

    cases2dim2 = TrabCases()
    for _ in range(10):
        cases2dim2.randomWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX)
        cases2dim2.tournamentWithOnepoint(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX)
        cases2dim2.randomWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX)
        cases2dim2.tournamentWithMultipoints(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX)
    cases2dim2.npArray()
    
    # Print results
    def print_results(cases, description,name):
        nameChange = name.split(":")
        f = open(f"Caso{nameChange[0]}.txt", "a")
        f.write(f"============== {description} =============="+"\n")
        f.write("Randomico com um ponto:"+"\n")
        f.write("Media fitness de 10 iteracoes: " + str(np.sum(cases.randomWithOnepointFitness) / 10) +"\n")
        f.write("Melhor fitness de 10 iteracoes: " + str(np.min(cases.randomWithOnepointFitness))+"\n")
        
        f.write("Torneio com um ponto:"+"\n")
        f.write("Media fitness de 10 iteracoes: " + str(np.sum(cases.tournamentWithOnepointFitness) / 10)+"\n")
        f.write("Melhor fitness de 10 iteracoes: " + str(np.min(cases.tournamentWithOnepointFitness))+"\n")

        f.write("Randomico com multiplos pontos:"+"\n")
        f.write("Media fitness de 10 iteracoes: " + str(np.sum(cases.randomWithMultipointsFitness) / 10)+"\n")
        f.write("Melhor fitness de 10 iteracoes: " + str(np.min(cases.randomWithMultipointsFitness))+"\n")

        f.write("Torneio com multiplos pontos:"+"\n")
        f.write("Media fitness de 10 iteracoes: " + str(np.sum(cases.tournamentWithMultipointsFitness) / 10)+"\n")
        f.write("Melhor fitness de 10 iteracoes: " + str(np.min(cases.tournamentWithMultipointsFitness))+"\n")
        f.write("=========================================="+"\n")
        f.close()

        

        

    print_results(cases10dim1, "Casos Convexo e 10 dimensoes",NAME_CONVEX)
    print_results(cases10dim2, "Casos nao Convexo e 10 dimensoes",NAME_NCONVEX)
    print_results(cases2dim1, "Casos Convexo e 2 dimensoes",NAME_CONVEX)
    print_results(cases2dim2, "Casos nao Convexo e 2 dimensoes",NAME_NCONVEX)

    # Plot heatmaps
    plot_heatmap_with_points(func=FUNC_CONVEX, cases=cases2dim1,name = NAME_CONVEX)
    plot_heatmap_with_points(func=FUNC_NCONVEX, cases=cases2dim2,name = NAME_NCONVEX)

    probCrossVect = [0.7,0.75,0.8,0.85,0.9,0.95]
    k_wayTournamentVect = [0.2,0.3,0.4,0.5]

    casesTestCase1Dim2 = TrabCases()
    casesTestCase2Dim2 = TrabCases()

    for i in range(len(probCrossVect)):
        for j in range(len(k_wayTournamentVect)):
            casesTestCase1Dim2.testCaseTournament(term=TERM_DICT,problem_dict=PROBLEM_DICT_CONVEX,pc=probCrossVect[i],k_way=k_wayTournamentVect[j])
            casesTestCase2Dim2.testCaseTournament(term=TERM_DICT,problem_dict=PROBLEM_DICT_NCONVEX,pc=probCrossVect[i],k_way=k_wayTournamentVect[j])
    casesTestCase1Dim2.npArray()
    casesTestCase2Dim2.npArray()

    plot_heatmap_with_points_testCase(func=FUNC_CONVEX, cases=casesTestCase1Dim2,name = NAME_CONVEX)
    plot_heatmap_with_points_testCase(func=FUNC_NCONVEX, cases=casesTestCase2Dim2,name = NAME_NCONVEX)


def run_test_cases2():
    
    test1 = TrabCases()
    test2 = TrabCases()
    p_aValues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    for i in range(len(p_aValues)):
        test1.csaCase(term=TERM_DICT,problem_dict=PROBLEM_DICT_5,p_a=p_aValues[i])
        test2.csaCase(term=TERM_DICT,problem_dict=PROBLEM_DICT_9,p_a=p_aValues[i])
    p_aValue1 = np.argmin(test1.csaCaseFitness)
    p_aValue2 = np.argmin(test2.csaCaseFitness)

    cases2dim1 = TrabCases()
    for _ in range(10):
        cases2dim1.testCaseTournament(term=TERM_DICT,problem_dict=PROBLEM_DICT_5, pc=0.9,k_way=0.5)
        cases2dim1.csaCase(term=TERM_DICT,problem_dict=PROBLEM_DICT_5,p_a=p_aValues[p_aValue1])
    cases2dim1.npArray()

    cases2dim2 = TrabCases()
    for _ in range(10):
        cases2dim2.testCaseTournament(term=TERM_DICT,problem_dict=PROBLEM_DICT_9, pc=0.9,k_way=0.5)
        cases2dim2.csaCase(term=TERM_DICT,problem_dict=PROBLEM_DICT_9,p_a=p_aValues[p_aValue2])
    cases2dim2.npArray()

    plot_heatmap_with_points_CaseWithCSA(func=FUNC_5, cases=cases2dim1,name = NAME_5, p_aValue = p_aValue1)
    plot_heatmap_with_points_CaseWithCSA(func=FUNC_9, cases=cases2dim2,name = NAME_9, p_aValue = p_aValue2)



if __name__ == "__main__":
    # run_test_cases()
    run_test_cases2()


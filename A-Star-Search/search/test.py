import subprocess
import sys


def test1():
    file_output = open('output.txt', 'w', encoding='utf8')

    subprocess.Popen("python pacman.py -l tinyMaze -p SearchAgent -a fn=dfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l tinyMaze -p SearchAgent -a fn=bfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l tinyMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -q",
                     stdout=file_output).wait()
    file_output.write("=======================================\n")

    subprocess.Popen("python pacman.py -l smallMaze -p SearchAgent -a fn=dfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l smallMaze -p SearchAgent -a fn=bfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l smallMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -q",
                     stdout=file_output).wait()
    file_output.write("=======================================\n")

    subprocess.Popen("python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -q",
                     stdout=file_output).wait()
    file_output.write("=======================================\n")

    subprocess.Popen("python pacman.py -l bigMaze -p SearchAgent -a fn=dfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -q", stdout=file_output).wait()
    file_output.write("=======================================\n")
    subprocess.Popen("python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic -q",
                     stdout=file_output).wait()
    file_output.write("=======================================\n")

    file_output.close()


def test2():
    subprocess.Popen(
        "python pacman.py -l foodSearch -p SearchAgent -a fn=astar,prob=FoodSearchProblem,heuristic=foodHeuristic").wait()


def test3():
    subprocess.Popen("python pacman.py -p MinimaxAgent -l testClassic -a depth=3").wait()
    subprocess.Popen("python pacman.py -p AlphaBetaAgent -l testClassic -a depth=3").wait()


if __name__ == '__main__':
    if sys.argv[1] == 'test1':
        test1()
    elif sys.argv[1] == 'test2':
        test2()
    elif sys.argv[1] == 'test3':
        test3()

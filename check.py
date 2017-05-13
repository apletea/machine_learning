import numpy as np



def main():
    A = np.eye(3)
    B = np.eye(3)
    AB = np.vstack((A,B))
    print(AB)
main()
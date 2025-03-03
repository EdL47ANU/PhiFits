import os
import sys
import subprocess

def run_simulation(input_file):
    subprocess.run(['phi_v3.1.6_gfortran.x', input_file],
                   stdout = subprocess.DEVNULL,
                   stderr = subprocess.DEVNULL)
def main():
    input_file = sys.argv[1]
    run_simulation(input_file)
if __name__ == "__main__":
    main()

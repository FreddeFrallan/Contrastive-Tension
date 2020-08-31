import os

gpu = input("GPU:")
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

from ContrastiveTension import Training

if __name__ == '__main__':
    Training.main()

    print("\nWork Complete!")

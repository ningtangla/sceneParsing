import pandas as pd

def main():
    originalDf = pd.read_csv('image/new_informationcontent_conditionalentropy_numTreeSpace_depthFlattestTree_expt6_sub16.csv')

    for key, dataDf in originalDf.groupby(['sub']):
        if key!=0:
            


if __name__ == "__main__":
    main()



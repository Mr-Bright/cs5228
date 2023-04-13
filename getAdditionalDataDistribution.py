# Author Oliver
# Date 2023/4/6 23:24
import pandas as pd
import matplotlib.pyplot as plt

def getAddtionalDataDistribution(env,col_name):
    dataset = pd.read_csv('Preprocessing/processed_' + env + '.csv')

    col_data = dataset[col_name]

    # Calculate the range of the column data
    column_range = col_data.max() - col_data.min()

    # Print the range of the column data
    print(f"The range of values in the '{col_name}' column is: {column_range}")

    # 计算每个 0.1 数值出现的次数
    counts = {}
    for value in col_data:
        key = round(value, 1)
        if key not in counts:
            counts[key] = 1
        else:
            counts[key] += 1

    data_dict = {}
    print(data_dict)

    # 显示每个 0.1 数值出现的次数
    for key, value in counts.items():
        #print(f"数值 {key} 出现了 {value} 次")
        data_dict[key] = value

    sorted_dict = dict(sorted(data_dict.items()))

    print(sorted_dict)

    # Get the keys and values from the dictionary
    keys = list(sorted_dict.keys())
    values = list(sorted_dict.values())

    # Create a bar graph from the keys and values
    plt.bar(range(len(keys)), values, tick_label=keys)

    # Add labels to the graph
    plt.xlabel('Number')
    plt.ylabel('Amount')
    name = env+' '+col_name+' Graph'
    plt.title(name)

    # Save the figure as a PNG file
    plt.savefig('figure/'+name+'.png')

    # Show the graph
    plt.show()

def getTwoDatasetFigure(feature):
    getAddtionalDataDistribution('train', feature)
    getAddtionalDataDistribution('test', feature)

if __name__ == '__main__':
    getTwoDatasetFigure('sg-primary-schools_lessHalfKMNum')
    getTwoDatasetFigure('sg-primary-schools_half2OneKMNum')
    getTwoDatasetFigure('sg-primary-schools_one2ThreeKMNum')
    getTwoDatasetFigure('sg-commerical-centres_lessHalfKMNum')
    getTwoDatasetFigure('sg-commerical-centres_half2OneKMNum')
    getTwoDatasetFigure('sg-commerical-centres_one2ThreeKMNum')
    getTwoDatasetFigure('sg-secondary-schools_lessHalfKMNum')
    getTwoDatasetFigure('sg-secondary-schools_half2OneKMNum')
    getTwoDatasetFigure('sg-secondary-schools_one2ThreeKMNum')
    getTwoDatasetFigure('sg-shopping-malls_lessHalfKMNum')
    getTwoDatasetFigure('sg-shopping-malls_half2OneKMNum')
    getTwoDatasetFigure('sg-shopping-malls_one2ThreeKMNum')
    getTwoDatasetFigure('sg-train-stations_lessHalfKMNum')
    getTwoDatasetFigure('sg-train-stations_half2OneKMNum')
    getTwoDatasetFigure('sg-train-stations_one2ThreeKMNum')
    getTwoDatasetFigure('sg-gov-markets-hawker-centres_lessHalfKMNum')
    getTwoDatasetFigure('sg-gov-markets-hawker-centres_half2OneKMNum')
    getTwoDatasetFigure('sg-gov-markets-hawker-centres_one2ThreeKMNum')

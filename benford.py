import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import stats

# data source: https://ourworldindata.org/coronavirus-source-data


def setup(csv_filename, country):
    df = pd.read_csv(csv_filename)
    df = df.loc[: , ["location", "date", "new_cases" ]]
    df = df[df["location"] == country]
    # Data prep
    df = df[df["new_cases"].notna()] # remove not a number-rows
    df = df[df["new_cases"] > 0]
    df.drop_duplicates()
    print(df.head())
    return df

def statistics(df):
    print("Dataset size", len(df["new_cases"].tolist()))    

def calculate_first_digit(df):
    new_cases = df["new_cases"]
    first_digit = []
    for row in df["new_cases"]: # get first digit
        try:
            first_digit.append(int(str(row)[:1]))
        except:
            first_digit.append(0)
            print(row)
    df["first_digit"] = first_digit
    df = df.drop(df[df.first_digit <= 0].index) # drop rows with 0 values
    n = len(df["first_digit"].tolist())
    count_first_digit = df["first_digit"].value_counts(sort=False)#count number of 1's, 2's, 3's and so on
    count_first_digit.to_frame().to_numpy()
    total_count = count_first_digit.sum() # number of numbers in list. Equal to len(df["first_digit"].tolist())
    percentage = []
    for elem in count_first_digit:
        p = float("{:.4f}".format( elem / total_count))
        percentage.append(p)
    x = np.linspace(1,9,9)
    percentage = dict(zip(x, percentage))
    return df, percentage

def calculate_first_two_digits(df):
    first_two = []
    for row in df["new_cases"]:
        temp_int = int(row*10)
        first_two.append(int(str(temp_int)[:2]))
    df["first_two"] = first_two

    count_first_two = df["first_two"].value_counts(sort=False)[1:]
    print(count_first_two)
    count_first_two.to_numpy()
    total_count = count_first_two.sum()
    percentage = []
    for elem in count_first_two:
        percentage.append(float("{:.4f}".format( elem / total_count)))
    return df, percentage

def plot_figure(percentage, perfect_benford):
    _x = np.linspace(1, 9, 9)
    plt.plot(_x,percentage, label="first digit benford") # calculated perfentage
    plt.plot(_x, list(perfect_benford.values()), label="perfect benford")
    plt.xlabel("Digits")
    plt.ylabel("Percentage")
    plt.legend()
    plt.show()

def get_perfect_benford():
    x = np.linspace(1,9,9)
    y = [0.31, 0.176, 0.125, 0.097,0.079, 0.067, 0.058, 0.051, 0.046]
    return dict(zip(x,y))

def pearson_coefficient(list_a, list_b):
    assert (len(list_a) != 0)
    assert (len(list_b) != 0) # list b is perfect benford
    sum_a = sum(list_a)
    sum_b = sum(list_b)
    mean_a = float(sum_a / len(list_a))
    mean_b = float(sum_b / len(list_b))
    list_mean_a = [(x - mean_a) for x in list_a]
    list_mean_b = [(y - mean_b) for y in list_b]
    numerator = sum(x * y for x,y in zip(list_mean_a, list_mean_b))
    denominator = math.sqrt(sum(x*x for x in list_mean_a) * sum(y * y for y in list_mean_b))
    if (denominator != 0):
        p_value = numerator / denominator
    else: 
        p_value = 0
    print("------ Pearson coefficient --------")
    print(p_value)
    return p_value

def mantissa_arc_test(list_a):
    """
        The mantissa arc test. 
        :parm list_a: df["new_cases"]
        :return: p-value and a plot
    """ 
    x_coordinates = [(math.cos(2*math.pi * (math.log10(x) % 1))) for x in list_a] # abscissa - x-coordinate for the mantissa
    y_coordinates = [(math.sin(2*math.pi * (math.log10(x) % 1))) for x in list_a] # ordinate

    x_nominator = sum(math.cos(2*math.pi * (math.log10(x) % 1)) for x in list_a)
    y_nominator = sum(math.sin(2*math.pi * (math.log10(x) % 1)) for x in list_a)
    x_coordinate = x_nominator / len(list_a) # Center of mass
    y_coordinate = y_nominator / len(list_a) # Center of mass

    L_squared = (x_coordinate)**2 + (y_coordinate)**2 
    p_value = 1 - math.exp(-L_squared * len(list_a))
    print("--------- p-value ---------")
    print(p_value)
    
    ''' Plotting '''
    plt.scatter(x_coordinates, y_coordinates)
    plt.plot(x_coordinate, y_coordinate, 'o', color="red") # center of mass
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()

### TODO: implement chi-squared
def chi_squared(df, perfect_benford):
    ''' 
    #observed, expected ....
    H0: status_que_hypothesis
    '''
    sample_size = len(df["new_cases"].tolist())
    benford_distribution = [sample_size*x for x in perfect_benford] # expected distribution
    count_first_digit = df["first_digit"].value_counts(sort=False)
    residual_squared = [math.pow(x-y, 2) / y for x,y in zip(count_first_digit, benford_distribution)]
    degrees_of_freedom = (9-1)*(2-1)
    p_value= stats.chi2.pdf(sum(residual_squared), degrees_of_freedom)
    print("chi squared p-value: ",p_value)
    #print("residual", residual)
    #print(benford_distribution) 
    #print("Not implemented yet")
    


def main():
    df = setup("owid-covid-data.csv","Belgium") #csv_filename, country
    statistics(df) # Get info about the dataset
    df, percentage = calculate_first_digit(df)
    print(df.head())
    chi_squared(df, list(get_perfect_benford().values()))
    pearson = pearson_coefficient(list(percentage.values()), get_perfect_benford())
    mantissa_arc_test(df["new_cases"].tolist())
    plot_figure(list(percentage.values()), get_perfect_benford())

if __name__ == "__main__":
    main()

import csv, random, math
import statistics as st

def loadCsv(filename):
    lines = csv.reader(open(filename, "r"));
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]] 
    return dataset

def splitDataset(dataset, splitRatio): 
    testSize = int(len(dataset) * splitRatio); 
    trainSet = list(dataset);
    testSet = []
    while len(testSet) < testSize:
    #randomly pick an instance from training data 
        index = random.randrange(len(trainSet)); 
        testSet.append(trainSet.pop(index))
    return [trainSet, testSet]

#Create a dictionary of classes 1 and 0 where the values are the #instacnes belonging to each class

def separateByClass(dataset): 
    separated = {}
    for i in range(len(dataset)):
        x = dataset[i]
        if (x[-1] not in separated): 
            separated[x[-1]] = []
        separated[x[-1]].append(x)
    return separated

def compute_mean_std(dataset):
    mean_std = [ (st.mean(attribute), st.stdev(attribute))
        for attribute in zip(*dataset)]; #zip(*res) transposes a matrix (2-d array/list) 
    del mean_std[-1] # Exclude label
    return mean_std

def summarizeByClass(dataset):
    separated = separateByClass(dataset);
    summary = {} # to store mean and std of +ve and -ve instances 
    for classValue, instances in separated.items():
    #summaries is a dictionary of tuples(mean,std) for each class value 
        summary[classValue] = compute_mean_std(instances)
    return summary

#For continuous attributes p is estimated using Gaussion distribution 
def estimateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))) 
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, testVector):
    p = {}
    #class and attribute information as mean and sd
    for classValue, classSummaries in summaries.items():
        p[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = testVector[i] #testvector's first attribute #use normal distribution
            p[classValue] *= estimateProbability(x, mean, stdev);
    return p


def predict(summaries, testVector):
    all_p = calculateClassProbabilities(summaries, testVector)
    bestLabel, bestProb = None, -1
    for lbl, p in all_p.items():#assigns that class which has he highest prob 
        if bestLabel is None or p > bestProb:
            bestProb = p
            bestLabel = lbl
    return bestLabel

def perform_classification(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]: 
            correct += 1
    return (correct/float(len(testSet))) * 100.0

dataset = loadCsv('C:\\Users\\Gunjan\\Desktop\\ML_Dataset\\NaiveBayse5\\dataset5.csv');
print('Pima Indian Diabetes Dataset loaded...') 
print('Total instances available :',len(dataset))
print('Total attributes present :',len(dataset[0])-1)

print("First Five instances of dataset:") 
for i in range(5):
    print(i+1 , ':' , dataset[i])

splitRatio = 0.2
trainingSet, testSet = splitDataset(dataset, splitRatio) 
print('\nDataset is split into training and testing set.')
print('Training examples = {0} \nTesting examples = {1}'.format(len(trainingSet),
len(testSet)))
summaries = summarizeByClass(trainingSet);
predictions = perform_classification(summaries, testSet)

accuracy = getAccuracy(testSet, predictions)
print('\nAccuracy of the Naive Baysian Classifier is :', accuracy)

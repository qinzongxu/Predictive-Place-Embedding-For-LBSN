you need to train location embedding and adjust with lambda in program. Here is the paramter explain:
a. train1, the graph you want to embed into vector
b. train2, the graph you want to embed into vector
c. train3, the graph you want to embed into vector
d. intial, the initial location embedding
e. output, output file
f. size, the size of vector
g. order, the type of the model, default is 2
h. negative, number of negative examples; default is 5
i. samples, set the number of training samples as <int>Million; default is 1
j. threads, the thread of training

Lambda should be adjusted according to the data.
# Reading: The spelled-out intro to language modeling: building makemore
# Code: build_makemore_yay.ipynb

- We will be using a dataset of names to create generated names using neural network by training nn on our dataset
- This will be a character level language model - MakeMore
 
# Process
- Load the dataset
- [Learn about the dataset: Total len, shortest and longest entry], how much information we are getting from each entry.
- Start with building a bigram LM -> Using a given character build the next character -> Basically look at prev char to predict the next one.
- While building the bigram LM -> You do not only use 2 chars from an entry but there is extra information such as <start> and <end>
- For biagrams -> the statics is counting to understand the combinations. "ab occured x times"
- Get the counts for the data
- [Analyse the map of counts and biagram] - most likely bigrams
- Get this data into 2d array instead of dict as this is more convinent. To design the 2d array use pytorch
- After creating this 2d array, we can use matplot lib to viz better but optional.
- See how start and end signal clutter and create additional rows and cols in the matrix which is not really providing any info as start cannot be after any char and end cannot be before any char
- Optimise for start and end -> "."
- This concludes the data which we can use to sample from the bigram char LM

- To sample for first char, get the first row and normalise to generate probability distribution
- Then sample from the distribution - Torch multinominal - this will draw samples from the matrix
- Loop this sample until end and that is prob dist trained model generating names
- Instead of normalising everytime, create a normalised array. (KeepDim=True)

- At this point we want to evaluate the model - model quality same as micrograd. For this we can use training data and the prob dist.
- One way to analyse is to see what can be prob dist if everything has same weight
- This is probalbly give us a matrix (1Xx) meaning prob for each entry in the training set.
- We need to now create one number to define the model quality - LIKELHOOD
- Likehood is prod of all numbers but it is very small therefore we use log of likelehood.
- Log(a*b*b) = loga + logb + logc = Loglikelihood
- -inf < loglikelihood < 1
- Because the number is large and -ve we will see something like -36 and because it can jump alot during training not really good therefore
we can use -ve of this so the number will be 0 to inf and the lower the better and we can also average it which give a really good loss fn.

# Goal: Maximize likelihod of the data wrt model parameters (Statstical modeling)
# Model parameters: For the biagram language model this is the probablity distribution
# Thse model parameter will be trained to maximize the likelihood.

- Handle endge case for loss function as some values can be 0 of mul can become inf
- Do model smoothing -> Fake count +=1 the more you add the more uniform you will have and the less will be more peak.

# At this point we want to cast this into NN
- Create a training set of biagrams
- Two list: Input and targets (x, y) -> They are characters but inside the list we insert stoi
- These list should be tensor for NN purpose
- These values should be int to make sense but you cannot put int into a NN because you need to do matrix mul
- We can use one-hot encoding for this and convert it into NX27 -> Encode int into vectors
- Float can feed into NN
- Now time to insert a NN
- Define weight torch.rand(27, 1)
- Xenc @ W 
- Now we add 27 neurons
- Define weight torch.rand(27, 27) -> This will become 1st layer
- We will keep only 1 layer
- To understand the response of NN -> We exp() them. As this give us something which is eq to counts as they are +ve -> This is called logits exponentiated
- Now we get logits, eq counts and the prob by normatilisng it. Just like we did without NN
# The method of exponentiate and then normalize is called SoftMax
# In general, output if neural network can be converted into prob dist using softmax
- After this we work on loss function improvement using back propogation
- Implement calculation of loss function LogA+LogB+...
- Now we will implement backpropogation to min the loss -> Gradient decent algorithm 



### Assignment 5: Instances and networks.

#### Due Monday, May 17, 11:59pm.

100 points.

Question 1. (33%) Distance-based k-nearest neighbor. For this question, you'll implement k-Nearest Neighbor, using distance weighting, and test its performance on two datasets I've provided.
Both are classic ML datasets. The first is the Iris dataset, and the second is the Wine dataset.
   
I've provided an updated version of readARFF that should properly handle these files. As before, it reads the data into a pandas dataframe.

a. To begin, you'll want to write a distance function that can compare two instances (probably represented as pandas Series) and compute their Euclidean distance, just using the fesatures, not the classification. 

b. Next, write a function called majorityClassify. It should take two inputs: a dataframe of training data, and an instance to be classified.
It should compute, for each possible class, the distance-weighted vote for that class by iterating through the elements in the dataframe, computing their distance to the new instance,
and then updating a counter. (You can also do this in a functional style, using a list comprehension or pandas' select operator). Return the classification with the most votes.

c. Last, create a wrapper function to test this. It should take as input an ARFF filename, read in the data,
and do 5-fold cross-validation - collect 80% as your model, and then call majorityClassify on the other 20%, and compute the accuracy. I would 
strongly encourage you  to re-use or refactor your code from assignment 3. 

I should be able to call your runner and it should do the cross-validation and then print out the accuracy for each run as well as the average accuracy.

Question 2. (33%) In this question, you'll get some exposure to setting up and solving a basic deep learning problem using TensorFlow. This question is less about writing your own code and more about working through tutorials, figuring out how to get things running, and making modifications. 

TensorFlow can be run completely within the browser using CoLab. If you prefer, you are welcome to install it on your machine (or in a cloud instance)  for greater efficiency.

For submission, please add either Python files or a Jupyter notebook to your repository. 

First, work through this tutorial to get started with TensorFlow: https://www.tensorflow.org/tutorials/keras/classification

Next, work through the CNN tutorial here: 
https://www.tensorflow.org/tutorials/images/cnn

Last, work through the Image Recognition tutorial here:
https://www.tensorflow.org/tutorials/images/classification

These tutorials are great, but it's easy to wind up just clicking on things and not really absorb the material. Add a section to the bottom of your code
that answers the following questions. You may need to poke around the TensorFlow site a little bit to find them.

What is a MaxPooling2D layer? What's it do?

What's Adam? 

What does CategoricalCrossEntropy mean?

In the CNN example, what does the Flatten layer do?

In the CNN example, what does the Dense layer do? 

In the CNN example, why does the height and width get smaller for each convolutional layer?


Question 3. (33%) [(from Russell and Norvig, 18.27)](https://aimacode.github.io/aima-exercises/concept-learning-exercises/ex_27/)

Consider the following set of training examples A1-A14, each with six inputs X1-X6 and one
target output:

<table>
<tr><td>Example</td><td>A1</td><td>A2</td> <td>A3</td> <td>A4</td> <td>A5</td> <td>A6</td> <td>A7</td> <td>A8</td> <td>A9</td> <td>A10</td> <td>A11</td> <td>A12</td> <td>A13</td> <td>A14</td>      </tr>
<tr><td> X1  </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 0 </td></tr>
<tr><td> X2 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 1 </td></tr>
<tr> <td>	X3  </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td></tr>
<tr><td>	X4  </td><td> 0 </td><td> 1  </td><td> 0  </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 0 </td><td> 1 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 1 </td></tr>
<tr><td>	X5  </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 0 </td></tr>
<tr><td>	X6  </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 0 </td></tr>
<tr><td>	Output   </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 0 </td></tr>
</table>

Trace the perceptron learning algorithm on this data and show the final weights. Assume that alpha = 0.1.
(if you would prefer to write a program to do this, that's fine too.)


4. (CS686 students). One of the enabling technologies that initially made deep learning so effective was ImageNet. 

To begin, [please read this paper](http://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf). 
 This was written when ImageNet first launched, and describes the architecture.
    
    
[This 2017 presentation](https://imagenet.stanford.edu/static_files/files/imagenet_ilsvrc2017_v1.0.pdf) summarizes the impact that ImageNet had after its introduction. 

Prepare a summary that addresses the contributions of ImageNet, in particular:

- ImageNet was the first large-scale publicly-available dataset of labeled images. What are the advantages of making ImageNet free and publicly available?

- ImageNet was built in alignment with WordNet. What is WordNet? Why was it helpful to use WordNet in labeling images?

- ImageNet demonstrates a change in the approach to AI, away from hand-crafted solutions to the use of large-scale datasets to tackle difficult problems. Where else have we seen this tension and shift in our class?


  

    

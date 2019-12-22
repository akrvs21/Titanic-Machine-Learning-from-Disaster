### Titanic: Machine Learning from Disaster  

### Part 1 

Define the competition requirements: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

The train dataset and test dataset for this competition are provided by Kaggle.  It's exploratory data analysis so I used Pandas for loading, testing and training the data. 

Pandas introduction:  

In computer programming, **pandas** is a software library written for the Python programming language for data manipulation and analysis. In  particular, it offers data structures and operations for manipulating  numerical tables and time series.

![image-20191221151030010](\img\image-20191221151030010.png)

Above import Pandas and create a data frames for train and test so we can visualize them using Pandas.  



![image-20191221153212483](img\image-20191221153212483.png) 

With `train.head()` we can retrieve all the features that we have in our `train.csv` file. I will clarify some of them here:  

- `Pclass` - there is first second and third classes. It's like in airplane,  business class and economy class . Here 1 is expensive class and 3 is cheapest class.  

- `Survived` 0 means the passenger is died, 1 means survived.   

- `SibSp` - means sibling and spouse. If it's 0 that means that a passenger doesn't have sibling neither spouse.  

- `Parch` - means parents and children. Same, if it's 0 then doesn't  have both. 

Other columns I think pretty straight forward such as Name, Sex etc.

We can see that some values have `NaN`, which stands for `"Not a number"`. We can understand it as a missing values or `NULL`, so we have to fill this fields using `Feature Engineering` techniques.   



![image-20191221155215781](img\image-20191221155215781.png) 

 **PANDAS** allow us to see that `train.csv` has passengers and 12 features and `test.csv` has 11 features, it doesn't have the Survived column.

  

![image-20191221155748531](img\image-20191221155748531.png)

Here we can see that number of passengers is 891 and number of Age is 714 so we definitely have some missing values in Age. 



![image-20191221160516985](img\image-20191221160516985.png)

Similar for test.csv



To visualize the data, I use `seaborn` for categorical features. 

![image-20191221160946969](img\image-20191221160946969.png)

Above is the function for the chart. Is a get one feature as a parameter and going to give two bar charts, one is for "Survived" another for "Dead". 



![image-20191221161331641](img\image-20191221161331641.png)

As we can see the chart, the females (blue) survived most and male less. 



![image-20191221161513852](img\image-20191221161513852.png)

And here we can see that people from economic class(3d class or green) died more than the First class passengers (blue). 



![image-20191221161749799](img\image-20191221161749799.png)

Here we observe that if a passenger were alone, without siblings and a spouse, he more likely to die. 



### Part 2

In this part we are going to do some `feature engineering` to fill up the missing fields. Second, to change some values to numerical values, such as male and female, because for ML classifier it's easier to work with 0 and 1s. And also delete not important features, for example `Name`, because this feature doesn't give us any valuable information, instead we can remove the name and retrieve the partial information from `Name` field such as *Mr*, *Miss*, *Mrs*. So we can know if it's male, female, married or not, and combine it in one new feature `Title`. 

Combine train and  test dataset  and extract only the title from the Name field 

![image-20191221165549404](img\image-20191221165549404.png)

Extracted above titles and  combine them as follows: 

`Mr`

`Miss` 

`Mrs `

`Others`  

And use mapping to change these title as a numeric value (0, 1, 2, 3)

![image-20191221165933086](img\image-20191221165933086.png)

![image-20191221170152134](img\image-20191221170152134.png)

Now we have new column Title.  I won't show for test dataset but it's same, because I did combine them. 



![image-20191221170357427](img\image-20191221170357427.png)

From out chart we can see that `Mr` less survived than `Miss` or `Mrs`. Similar picture we saw  for `Sex` comparison.

Then I drop the `Name` field from datasets

![image-20191221170637470](img\image-20191221170637470.png)



And change the text to numeric values , 0 for **male** and 1 for  **female**

![image-20191221170726345](img\image-20191221170726345.png)



And the last we have to fill the missing fields in Age. The technique is to get medium value of the Title's age, instead of using the average age of the whole Titanic passengers. 

![image-20191221171459867](img\image-20191221171459867.png)

and if we draw the chart, we can see that approximately 17 or 18 years old passengers have a high chance to survive

![image-20191221171829200](img\image-20191221171829200.png)

and 25 year old above have a high chance to die. 



We can get a closer look in some range of age with `xlim`.

![image-20191221172217193](img\image-20191221172217193.png)

Below I do the binning, or converting numerical age to categorical variables. Because we have a lot of passengers and all of them have different age, it's hard to see in chart all that information. So we divide into some categories, such as: child = 0, young = 1, adult = 2, mid-age = 3, old  = 4

![image-20191221172639556](img\image-20191221172639556.png)

![image-20191221173429948](img\image-20191221173429948.png)

For Embarked feature I'm filling with S value because as we can observe it's the most from our chart. 

![image-20191221173754819](img\image-20191221173754819.png)

The `Fare` also has some missing values. So we have to fill them. Because the `Fare` feature is very related to `Pclass` so I'm going to find the median value of a `Pclass` and fill out the missing value for `Fare`. 

![image-20191221174258985](img\image-20191221174258985.png)

And similar to age technique do the binning :

![image-20191221174408754](img\image-20191221174408754.png)

And again if the passenger is alone there is a high chance to die, and if he/she has sibling  or spouse etc. there is  higher chance to survive 

![image-20191221174832354](img\image-20191221174832354.png)

This code is mapping the family size with some numeric value for simplicity 

![image-20191221175056191](img\image-20191221175056191.png)

![image-20191221175232993](img\image-20191221175232993.png)

Above I drop the ticket information, because it's pretty useless. And `SibSp` and `Parch` feature because I did combine them in `FamilySize` field. 

### Part 3

In this section, I will build `Multiple classifier` to predict Titanic test dataset. Also do the `Cross Validation`  `kFold`,  which allows to validate all over the data and help us to find best fit model. 

Modeling

![image-20191221183009444](img\image-20191221183009444.png)

For this lab, I used 5 popular classifiers:

1. kNN algorithm - a type of **supervised machine learning** algorithm. The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

   ![image-20191221183057243](img\image-20191221183057243.png)

   n_splits = 10, so we divide the train data into 10 fold, first 9 folds has 89 rows and last fold has 90 row. Then validate 10 rounds, different fold for each round. After finishing a round, average accuracy and pick the best accuracy model for this test. Others are similar. 

   2. Decision Tree - *A decision tree is drawn upside down with its root at the top.* In the image below, the bold text in black represents a condition/**internal node**, based on which the tree splits into branches/ **edges**. The end of the branch that doesn’t split anymore is the decision/**leaf**, in this case, whether the passenger died or survived, represented as red and green text respectively. This is very basic illustration for demonstration purpose and doesn't represent all the actual features. 

      

      ![image-20191221183506185](img\image-20191221183506185.png)

      ![image-20191221183635562](img\image-20191221183635562.png)

      3. Random Forest Tree -  Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction (see figure below).
   
         ![image-20191222150540736](img\image-20191222150540736.png)
      
         
      
         4. Naive Bayes algorithm - to understand the Naive Bayes classifier we need to understand the Bayes theorem. So let’s first discuss the Bayes Theorem. Bayes theorem named after Rev. Thomas Bayes. It works on conditional **probability**. Conditional probability is the probability that something will happen, ***given that something else\* has already occurred**. Using the conditional probability, we can calculate the probability of an event using its prior knowledge.
      
         Below is the formula for calculating the conditional probability.
      
         ![\textrm{P(H \textbar E) = }  \frac{\textrm{ P(E \textbar H) * P(H)}} {\textrm{P(E)}}](https://s0.wp.com/latex.php?latex=%5Ctextrm%7BP%28H+%5Ctextbar+E%29+%3D+%7D+%C2%A0%5Cfrac%7B%5Ctextrm%7B+P%28E+%5Ctextbar+H%29+%2A+P%28H%29%7D%7D+%7B%5Ctextrm%7BP%28E%29%7D%7D&bg=ffffff&fg=000&s=0)
      
         **where** 
      
         - P(H) is the probability of hypothesis H being true. This is known as the prior probability.
      
         - P(E) is the probability of the evidence(regardless of the hypothesis).
      
         - P(E|H) is the probability of the evidence given that hypothesis is true.
      
         - P(H|E) is the probability of the hypothesis given that the evidence is there. 
      
           Naive Bayes is a kind of classifier  which uses the Bayes Theorem. It predicts membership probabilities for  each class such as the probability that given record or data point  belongs to a particular class.  The class with the highest probability  is considered as the most likely class. This is also known as **Maximum A Posteriori (MAP)**.
      
           #### The MAP for a hypothesis is:
      
           **MAP(H)**
            = max( P(H|E) )
            =  max( (P(E|H)*P(H))/P(E))
            = max(P(E|H)*P(H))
      
           P(E) is evidence probability, and it is used to normalize the result. It remains same so, removing it won’t affect.
      
           Naive Bayes classifier assumes that all the features are **unrelated** to each other. Presence or absence of a feature does not influence the  presence or absence of any other feature. We can use Wikipedia example  for explaining the logic i.e. 
      
           ![image-20191222152043780](img\image-20191222152043780.png)
      
           5. “Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges.  However, it is mostly used in classification problems. In this  algorithm, we plot each data item as a point in n-dimensional space  (where n is number of features you have) with the value of each feature  being the value of a particular coordinate. Then, we perform  classification by finding the hyper-plane that differentiate the two  classes very well (look at the below snapshot).
      
              [![SVM_1](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_1.png)](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_1.png)
      
              ![image-20191222153630503](img\image-20191222153630503.png)
      
              ![image-20191222153653365](img\image-20191222153653365.png)
      
              Since `SVM` has the best accuracy so I used it for testing. 
      
              ![image-20191222155256532](img\image-20191222155256532.png)
      
              Then generated new csv file and summited it on `Kaggle` platform.  
      
              
      
              Submission Result 
      
              ![image-20191222155853677](img\image-20191222155853677.png)
      
              
      
              
      




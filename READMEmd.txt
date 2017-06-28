# Deep-Scoring-Project  
About:
Research Project with Dr. Frank Hsu of Fordham University. The aim of this project is to take a set of scoring systems and get a better performance at each layer. We start with a set of scoring systems and combine them in interesting ways that aim to increase performance. No matter the alteration each new layer is its self a set of scoring system. If we can get a layer that improves the accuracy of the previous layer consistently, we can keep iterating it until it converges to some maximum. If it works as intended, this will essentially allow us to have an easy way to ensemble many sub-par models into a high performing one.


Original Model: Deepscoring.py
We've taken 5 given models created for a classification task, combined them using fusion methods, and use the notion of diversity to find the most diverse model and give it the highest weight while combining. After running this in part 1, we ended up with 4 different models all of which are really close to the highest performing model, but not better or equal. In part 2, we take the 4 outputs from part 1 and add the highest performing model from the original input and repeat the process. This yielded 4 models with the same accuracy as the highest performing model form the original. These results are a mixed bag, while it is converging to a perhaps maximum accuracy, it didn't out perform the best in the original 5 models. For now, we'll try to examine the errors and perhaps try this methodology with a different binary classification problem.


Combination of A and E:Combinaiton of A and D.py
There was an expectation from Dr. Hsu that the combination of A and D (being the most diverse) should yeild a result better than A alone but after testing it didn't.


Combinatoric Fusion: DS4.py, Results: Outperform A.xlsx
Measure precision with top 100, 200 and 300:
For each model Measure Rank Percision and Score Percision with top 100,200,300 as well as # of positive cases
Create average for each group in MixGroup Rank, and Mix Group Score and send them to be evaluated in evaluation function
email Dr.Hsu exactly how MixGroup Rank, Mix Group Score, and Average per Group works
Deliverable:
Function percision, that takes a ranking or scoring system and evaluates the performance at  top 100,200,300 as well as # of positive cases
Function checkPerGroupAvergeScore, It takes a set of scores or ranks and iteratively passes all combinations of that set to the Evaluation function
RESULTS:
From this test We found that
3 combo out performed at every percision
testing combo (0, 2), (0, 4), (0, 2, 3)
4 more combination of these models outperformed or
matched at every precision (0,1),(0, 1, 4),(0, 2, 4),(0, 3, 4)
Function checkPerGroupAvergeRankScore, It takes a set of ranks and iteratively passes all combinations of that set to the Evaluation functions
RESULTS:
From this test We found that
0 models outperformed or matched at #pos and P@300
and the same models outperformed or matched a at 200, and 100
Rank Performance varied since averaging ranks (especially that of A, would heavily skew results)
Next Steps:
Try the original model one with the 7 better models from the Combinatoric Fusion layer, or perhaps try to pass those models back into the combinatoric fusion function and see if it improves further. This is yet to be determined.

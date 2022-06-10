# FoondaMate-ML-Engineer-Coding-Challenge-001

# FoondaMate-ML-Engineer-Coding-Challenge-001
## NOTE: 
Code has been optimized to batch process multiple email texts at once! For more detailed explanation of the code, solution and appreciation of its result on test data, please, read through the python jupyter notebook in this repository. The csv files in this repository shows the results of the algorithm on test data.

## HOW TO RUN CODE
Run the cells of the jupyter notebook in this repository on google colab or a local notebook server to appreciate the performance and result of this solution.

Use these functions in your code by moving the email_classifier.py file (post download) to your code's current working directory. Then import all the functions from the file into your python code script. Pass the email text (or list of email texts) to `batch_classify()` as its argument during its call.

Run code on your local computer via CLI by either passing email text as string or a path to a text file that holds a list of email texts (one email text per line). 

Use `python email_classifier.py 'EMAIL TEXT'` or `python email_classifier.py 'TEXT_FILE_PATH' `.
It returns a string as a result or saves result (in the case of a text file path passed as argument) to an output text file called 'emails_classified.txt' in the current working directory.

**If spacy or numpy is not installed on your local computer, please do so using the `pip` command before running this code**

All codes are well commented and explained. **An explanation of how code works can be found below**.

## PROBLEM STATEMENT
Given an email from a student as a text that contains the stem 'share' and the word 'email', build a function that classifies the email into : (i) Student has shared (ii) Student wants to know if can share.

## SOLUTION DESCRIPTION
It can be modeled by checking whether these emails are structured as a Yes/No question or not, due to the structure of the english language . 
For example, 'Can I share your email?' is a Yes/No questions. In english language, several principles must be met before a Yes/No question can be formed.
They are:

-  The auxillary verb comes immediately before pronoun/subject in the sentence. For example, in the sentence 'CAN I share your email?'. This is by 
  far, the strongest determinant of a successful Yes/No question formation and consequently has the greatest weight in my solution.

- The use of punctuation marks '?'.

- Specific to this problem, is the use of the past tense of share -'shared'. 


## CODE IMPLEMENTATION
The 3 points above are used to craft a solution to the problem. **I use the inbuilt POS tagger in the spacy library to get the part of speech of the words in each email.**

The email_classifier.py file contains 5 functions for this task:

- `batch_classify()`: This is the main function. It takes input as a string of email text or list of email text. All other function calls exist within it.

- `batch_preprocess()`: Tokenizes emails (given that it contains the stem ' share' and the word 'email' and gets the part of speech tags using spacy library. 
  Then it gets the POS tags associated with it. 

- `batch_get_scores()`: This function checks if the auxillary verb comes immediately before pronoun in sentence(s). Assigns a score of 1 if it does else, 0.

- `batch_calc_total_score()`: It evaluates all the criteria (3 points above) and calculates a custom weighted average score using a predefined weights.  
  If none is given, a simple average of the 3 scores is calculated.
  
- `pad_sequences()`: This function is used to pad each email text to the same length for batch processing.
  
## HOW IT WORKS
`batch_classify` when called passes the email text(s) given as argument to `batch_preprocess` which first checks if the email contains the stem 'share' and the word 'email. If condition not met, function returns with an alert message. Any email text that doesnt meet this condition is removed in the case when a list of email text is passed as argument.

Subsequently, the email texts are tokenized and all POS tag information collected and stored in a tuple that includes the tokenized word itself. The tense form of the verb 'share' is collected and stored **(tense score)** as 1 if in past participle format or 0 if in base or present form. It also check if it contains a '?' stored **(punc score)** as 1 or 0 if there isnt. **This is done for each email messages if a list of email messages are provided**

This list of tuple is padded to equal length by `pad_sequences` and passed to `batch_get_scores` which checks if an auxillary verb comes immediately before a pronoun/subject. I take advantage of the multidimensional array to parallelize this condition using the numpy library. Now, if the email meets this condition, it returns a 1 and if not, it returns a 0 **(verb_before_pronoun score)**.

These 3 scores **(tense score, punc score, verb_before_pronoun score)** are then passed to `batch_calc_total_score` that uses a set of customizable predefined weights to calculate a weighted average score.

If this final score is >= 0.5, `batch_classify` function outputs a **'Student wants to know if your email can be shared.'** else it returns a **'Student has shared your email'** prompt.


The solution is tested on a list of 30 manually crafted emails (also containing emails that don't contain the stem 'share' and 'email') to evalutate performance (check the '.ipynb' python notebook and csv files for this). I also scale this list to around 12000 to test the batch processing function!





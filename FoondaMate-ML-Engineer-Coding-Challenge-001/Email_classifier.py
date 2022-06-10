
import sys


import spacy # install spacy when running on a local jupyter server.
import numpy as np
import pandas as pd


def batch_preprocess(sentences, batch_size=50):
  """
    It takes a list of sentences (or just a sentence) and converts into a list of list of tuples. Each tuple represents containing tokens and their tag properties.
    It also registers the tense form of the stem word 'share' in each email and the presence of the '?' punctuation mark.
    Args:
    - sentences. A text string or a list of strings (emails)
    Output:
    - appr_email_address. list of integer. contains the indexes of the emails that have the stem 'share' and the word 'email' in them.
    - sent_tokenized_list. list of list of tuples. list of emails tokenized along with their tag properties.
    - tense_scores. list of int. contains a score for each sentence (email) signifying if it contains the past participle of 'share'
    - has_punc. list of int. contains a score for each sentence (email) signifying if it contains a '?'.
    - sentence_length. list of int. contains the length of the tokens in each email or sentence.
  """

  if isinstance(sentences, str):# Check if the sentence argument is a string or a list of email text
    sentences = [sentences]# if yes, put the sentence into a list.

  sent_tokenized_list = [] # create list to store the tokenized form of the email texts along with its properties.
  tense_scores = [] # Create a list variable that scores the tense form of 'share' in the email. if in past participle {shared}, it appends 1 else it appends 0.
  appr_email_index = [] # Create a list that stores the indexes of all relevant emails (that contain the stem 'share' and the word 'email')
  sentence_length=[] # Create a list that stores the length of tokens in each sentence.
  punc_scores = [] # Create a list that stores 1 if a sentence contains a '?' and 0 if it does not.

  # Load our tokenizer object from spacy library
  nlp = spacy.load('en_core_web_sm')
  # For each email in the email_list:
  for ind, sentence in enumerate(nlp.pipe(sentences, n_process=-1, batch_size = batch_size, disable=["tok2vec",'ner', 'textcat', 'parser', "attribute_ruler"])):
    # if email contains the stem 'share' and the word 'email':
    if 'share' in " ".join([token.lemma_ for token in sentence]) and 'email' in " ".join([token.lemma_ for token in sentence]):
      sent_tokenized = [] # create list to store the tokens of the index sentence along with its properties. It stores each token and attributes as a tuple

      is_punc = 0 # initialize the is_punc function to 0

      tense_score = 0 # initialize the tense_score to 0. changes it to 1 when share is in its past participle format.

      # create the length variable to store the lenght of the tokens (words) in the each email text.
      length_of_tokens = 0

      # For each token in the index sentence
      for token in sentence:
        # continue updating the length variable:
        length_of_tokens += 1
        # If the lemma form of the token is 'share' (For example, lemma of sharing or shared == 'share'), append
        if token.lemma_ == 'share':
          # Append token to the list with its properties then set the tense_score variable to 1 if the tense form of 'share' is in past participle.
          sent_tokenized.append((token.lemma_, token.tag_, token.pos_, spacy.explain(token.tag_)))
          if spacy.explain(token.tag_) == 'verb, past participle':
            tense_score = 1

        elif token.lemma_ == '?':# if token is the '?', set the is_punc to 1, then append the token with its properties to the list
          is_punc = 1
          #sent_tokenized.append((token, token.tag_, token.pos_, spacy.explain(token.tag_))) # otherwise, append token and its properties to the list 
        else: # Else just append
          sent_tokenized.append((token, token.tag_, token.pos_, spacy.explain(token.tag_))) # otherwise, append token and its properties to the list
      
      # update all the lists created at the beginning of the function accordingly at each sentence (email text) level.
      sentence_length.append(length_of_tokens)
      tense_scores.append(tense_score)
      punc_scores.append(is_punc)
      appr_email_index.append(ind)
      sent_tokenized_list.append(sent_tokenized)
    else: # Else, skip the email
      continue

  # If there are relevant emails (that contain the stem 'share' and the word 'email') in the list of email text, return all the updated list
  if len(appr_email_index) != 0 : 
    return appr_email_index, sent_tokenized_list, np.array(tense_scores), np.array(punc_scores), sentence_length
  else: # Else return None signifying that all emails in the list are invalid for this operation
    return None


def batch_get_score(token_sent):
  '''
    Calculates a score (0 or 1) that represents whether the auxillary verb in each sentence comes before the pronoun.
    Args:
    - token_sent. list of list of tuples (list of list of tokens and corresponding tag properties)
    Output:
    - comes_before_pronoun. a list of scores : (0 or 1)
  '''
  # size == (batch_size, word_length, word_properties {4})
  token_sent = np.array(token_sent)

  # token_array.shape == (batch_size, word_length_per_sentence, word_properties*2)
  token_array = np.concatenate((token_sent,np.roll(token_sent, -1, 1)), axis= -1)

  # Check if uncontracted auxillary verb comes immediately before pronoun/subject. E.g 'CAN I send your email'
  # temp_size.shape == (batch_size, word_length_per_sentence, 2)
  temp = np.concatenate((np.any(np.all(token_array[:,:,[2,5]]== ['AUX','PRP'], axis=2), axis=1, keepdims=True),
                         np.any(np.all(token_array[:,:,[3,5]]== ['verb, modal auxiliary','PRP'], axis=2 ), axis=1, keepdims=True)),axis= -1) 
  
  #verb_before_pronoun_score.shape == (batch_size, 1)
  verb_before_pronoun_score = np.any(temp, axis=-1, keepdims=True)

  # Checks if contracted auxillary verb comes immediately before pronoun/subject. E.g 'I can share your email, CAN'T I'
  # token_array.shape == (batch_size, word_length_per_sentence, word_properties*3)
  token_array = np.concatenate((token_sent, np.roll(token_sent, -1, 1), np.roll(token_sent, -2, 1)), axis=-1)

  # temp.shape == (batch_size, word_length_per_sentence, 2)
  temp = np.concatenate((np.any(np.all(token_array[:,:,[2,5,6,7,9]]== ['AUX','RB', 'PART','adverb','PRP'], axis=2), axis=1, keepdims=True),
                         np.any(np.all(token_array[:,:,[3,5,6,7,9]]== ['verb, modal auxiliary','RB', 'PART',
                           'adverb', 'PRP'], axis=2), axis=1, keepdims=True)), axis= -1)
  
  # Check if each sentence have the contracted version or the uncontracted version of auxillary verb coming before the pronoun.
  # verb_before_pronoun_score.shape == (batch_size, 2)
  verb_before_pronoun_score = np.concatenate((verb_before_pronoun_score, np.any(temp, axis=-1, keepdims=True)), axis=-1)
  # verb_before_pronoun_score.shape == (batch_size, 1)
  verb_before_pronoun_score = np.any(verb_before_pronoun_score, axis=-1).astype('int')

  return verb_before_pronoun_score


def batch_calc_total_score(verb_before_pronoun_score, tense_score, punc_score, weight = [0.5, 0.3, 0.2]):
  '''
     Calculate the final score given 3 scores provided as positional arguments. It uses the weight parameter to calculate the final score if given or
     it calculates an average over 3 scores.
     Args:
     - verb_before_pronoun_score : list of integers. whether an auxillary verb comes before the pronoun. 1 if it does else 0.
     - tense_score : list of integers (0,1). represents whether an index sentence contains the past participle form of 'share' or not.
     - punc_score : list of intergers (0,1). represents whether an index sentence contains a '?' or not.
     - weight : list of 3 values. Modifiable. it's used to calculate a weighted average of the 3 arguments above.
    Output:
    - score. an array of calculated scores for each sentence (email)
  '''
  # If weight is given:
  if weight != None:
    # Calculate weighted average score using contents of the weight.
    score = verb_before_pronoun_score*weight[0] + (1 - tense_score) * weight[1] + punc_score * weight[2]
    return score
  # Else, calculate the average of the 3 scores.
  else:
    score = (verb_before_pronoun_score + (1 - tense_score) + punc_score)/3

  return score



def pad_sequences(sentences, max_length , pad_value= [('sp','sp','sp','sp')]):
  '''
    Pads all emails to the same token length using the value of the pad_value argument.
    Args:
    - sentences: list of strings. list of tokenized emails.
    - max_length: int. length to which to pad sequences.
    - pad_value = list of tuple. the value to use in padding the sequences.
    Output: 
    - sentences. list of sequences padded to equal length.
  '''
  # Go through each email and pad each token sequence to the same length with the provide pad_value argument using the max_legth as argument.
  for ind  in range(len(sentences)):
    sentences[ind] += pad_value * (max_length - len(sentences[ind]))
  return sentences


def batch_classify(sentences, batch_size=250,  weight = [0.5, 0.3, 0.2], return_score=True):
  '''
    Main function: processes the list of email text, passing it to all other functions then returns the result in string format.
    Args:
    - sentences: List of strings or str. list of emails or a one email in string format.
    - batch_size: int. number of emails to process at once.
    - weight: list of float. should contain list of weights to use for calculation.
    - return_score: Float. if calculated scores should be returned.
    Output:
    - A list of strings whose elements states either : 'Student wants to know if your email can be shared' or 'Student has shared your email'
  '''
  # Check if sentences is a string in the case of just one email passed.
  # Set the is_text variable True or False accordingly.
  if isinstance(sentences, str):
    is_text = True
  else:
    is_text = False

  # Call the batch_preprocess function and collect all the output.
  email_ids , sentences, tense_scores , punc_scores , sent_length = batch_preprocess(sentences, batch_size = batch_size)

  # if the list containing the indexes of all the relevant emails is empty, return the string
  if email_ids == None:
    return "The text provided do not contain the stem 'share' and the word 'email' and so, could not be processed."
  
  # Create a list to append the result of the algorithm
  result = []
  # Create a list to store all the scores for each of the email text.
  scores = []

  # pad each processed email in the list.
  sentences = pad_sequences(sentences, max(sent_length))

  # process and calculate score in batches (using the batch_size):
  for start in range(0, len(sentences), batch_size):
    # Get score that represents that the auxillary verb comes before the pronoun:
    verb_before_pronoun_score = batch_get_score(sentences[start: start+batch_size])
    # Calculate the final scores for each email in the list
    score = batch_calc_total_score(verb_before_pronoun_score, tense_scores[start: start+batch_size], punc_scores[start: start+batch_size], weight= weight)
    
    # Convert scores to output string results : 'Student wants to know if your email can be shared.' or 'Student has shared your email.'
    # Then append to the result list.
    result.extend(list(np.where(score >= 0.5, 'Student wants to know if your email can be shared.', 'Student has shared your email.')))
    # append scores to the scores list.
    scores.extend(list(score))
    # Set start to the the beginning of the next batch.
    start += batch_size

  # If one email was passed, return the result of the only element in the result list.
  if is_text:
    # if return score, return the calculated score of the email as well.
    if return_score:
      return result[0], scores[0]
    else:
      return result[0]
  else: # Else, return the list of relevant email ids , corresponding results +/- scores.
    if return_score:
      return email_ids, result, scores
    else: 
      return email_ids, result



if __name__ == '__main__':
  if isinstance(sys.argv[1], str):
    print(batch_classify(sys.argv[1]))
  else:
    try:
      with open(sys.argv[1], 'r') as file:
        emails = file.readlines()
      email_ids, results, scores = batch_classify(emails)
      with open('emails_classified.txt', 'w') as file:
        for id , result, score in zip(email_ids, results, scores):
          file.write(str(id+1) + '\t' + str(result) + '\t' + str(score))

    except:
      print('The argument provided is not in string (text) format or a valid path to a text (.txt) file.')
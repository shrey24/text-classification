# text-classification
A predictive model that can identify the class of medical condition, with high precision, from a given medical abstract. Keywords: Text pre-processing, kNN, F1 scoring metric for evaluation.

Medical abstracts describe the current conditions of a patient. Doctors routinely scan dozens or hundreds of abstracts each day as they do their rounds in a hospital and must quickly pick up on the salient information pointing to the patient’s malady. You are trying to design assistive technology that can identify, with high precision, the class of problems described in the abstract. In the given dataset, abstracts from 5 different conditions have been included: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions.

The goal is to develop predictive models that can determine, given a particular medical abstract, which one of 5 classes it belongs to. For this, I have implemented min-epsilon k-NN classifier.

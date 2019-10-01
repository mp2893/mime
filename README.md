# MiME: Multi-level Medical Embedding of EHR for Predictive Healthcare
![alt text](https://i.ibb.co/41jQ2C5/ehr-structure.png "Hierarchical structure of EHR")

MiME is a novel EHR embedding algorithm that takes into account the connections between diagnoses and corresponding treatments (as shown in figure above) when learning to represent each hospital visit. The visit representation vectors are then sequentially fed to an RNN, whose hidden layer represents an entire visit history of a patient. The current code only supports a binary prediction of a given patient record (i.e. feed the last hidden layer of RNN to a sigmoid-activated linear layer). It can be used for prediction tasks such as heart failure prediction, or mortality prediction.

#### Relevant Publication

MiME implements an algorithm introduced in the following [paper](http://papers.nips.cc/paper/7706-mime-multilevel-medical-embedding-of-electronic-health-records-for-predictive-healthcare):

	MiME: Multilevel Medical Embedding of Electronic Health Records for Predictive Healthcare
	Edward Choi, Cao Xiao, Walter F. Stewart, Jimeng Sun  
	NIPS 2018
  
#### Notice

Since MiME takes advantage of the graphical structure of visit records, you CANNOT use MiME when structure information is unknown (e.g. MIMIC-III does not tell you which diagnosis code led to ordering specific medications). In such cases, I recommend using [Graph Convolutional Transformer](https://arxiv.org/pdf/1906.04716.pdf) (GCT), which tries to learn the visit structure while performing predictions. (GCT code will be released soon)

#### Running MiME

**STEP 1: Installation**  

1. Install [python](https://www.python.org/), [TensorFlow](https://www.tensorflow.org/install). We use Python 2.7, TensorFlow 1.14. 

2. If you plan to use GPU computation, install [CUDA](https://developer.nvidia.com/cuda-downloads)

3. Download/clone the MiME code.

4. To test the code, use the sample data with the command `python mime.py ./sample ./ ./test_sample`.

**Step 2: Data Preparation**

In order to use your own data, you must format it in a specific manner. An input sequence must be a cPickled nested List with a default extension `.seqs`. The outermost List represents all patients, the next List a single patient, the next List a single visit, the inner most List a single Dx-Object. Dx-Object List has a format `[diagnosis_code, [medication_codes], [procedure_codes]]`, where the medication_codes and procedure_codes were ordered due to diagnosis_code.

An input sequence of a single patient with two visit records, for example, can be `[[[[0, [0, 1], []], [1, [0], [0, 2]]], [[2, [2], []]]]]`. In the first visit, this patient had two Dx-Objects. The first Dx-Object consists of diagnosis_code_0, medication_code_0, medication_code_1. The second Dx-Object consists of diagnosis_code_1, medication_code_0, procedure_code_0, and procedure_code_2. In the second visit, the patient has only one Dx-Object, which consists of diagnosis_code_2 and medication_code_2. 

Note that in order to convert your EHR data to the MiME format, you must know a priori the reason diagnosis code for each medication and procedure order (i.e. Medication "Tylenol" was ordered due to diagnosis "Headache"). If such information is unvailable, consider using [Graph Convolutional Transformer](https://arxiv.org/pdf/1906.04716.pdf), which is essentially a generalization of MiME.

As mentioned above, the current code only supports binary predictions. Therefore your labels must be a cPickled List consisting of 0s and 1s. The file extension by default is expected to be `.labels`.

**Step 3: Hyperparameters**

Aside from the usual embedding_size and rnn_size, there are parameters that must be set correctly, if you are using your own dataset.

* `num_dx`: Number of unique diagnosis codes in your data
* `num_rx`: Number of unique medication codes in your data
* `num_pr`: Number of unique procedure codes in your data
* `max_dx_per_visit`: Maximum number of diagnosis codes a patient can receive in a single visit. This is set for limiting memory usage, but it is also unreasonable to receive 100 different diagnosis codes at a single visit.
* `max_rx_per_dx`: Maximum number of medications that can be ordered due to a single diagnosis. This is set for limiting memory usage, but is is also unreasonble to receive 100 different medications for a single diagnosis.
* `max_pr_per_dx`: Maximum number of procedures that can be ordered due to a single diagnosis. This is set for limiting memory usage, but is is also unreasonble to receive 100 different procedures for a single diagnosis.
* `min_threshold`: Patients with the number of visits less than this number will be filtered out.
* `max_threshold`: Patients with the number of visits more than this number will be filtered out.
* `aux_lambda`: This is a trade-off variable between the main prediction task and the auxiliary prediction tasks.
* `train_ratio`: If you set this to 0.5, you will only use a 50% of the entire training set.
* `association_threshold`: This should always be 0.0. Otherwise, you will be filtering out patients with relatively "simple" visit structure (e.g. has only one Dx-Object with no medication code and procedure code).

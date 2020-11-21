## Process Raw Dataset

To obtain high-quality dataset, we process raw dataset in `/raw_dataset` as follows:

1. After we delete duplicate methods, we obtain 74489 methods from 151414 methods.

2. To generate a common dataset for all models(DeGraphCS, DeepCS, MMAN and UNIF), we delete those methods which do not generate AST and CFG. Then we obtain 59725 methods.

3. To make sure our dataset is high-quality, we constraint comments' length and quality, and the number of nodes in AST, CFG and VFG(Variable-based Flow Graph). 
After we delete those methods which do not meet our requirements, we obtain 41152 methods in `/preprocessed_dataset`.
 


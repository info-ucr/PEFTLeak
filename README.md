1. Run the file Adapter_attack.ipynb. That will call functions from the following files.
2. Transformer_Model_neuron.py contains the architecture with inserted adapter module
3. Design_Model_Adapter.py contains the code for malicious design of the parameters
4. Processing_v2.py contains the code for creating patch statistics' intervals according to public dataset
5. Recover_Adapter.py recovers the patches from adapter gradients
6. Clustering.py groups the recovered patches to create the entire images (motivated by Fowl et. al., 2023)

 
==================================
Incremental Slow Feature Analysis 
==================================

Author - Varun Raj Kompella. (https://varunrajk.gitlab.io)

This is a free software; you can redistribute it and/or modify it. 
The code is distributed in the hope that it will be useful.  

If you plan to use this code in your research
please cite: V. R. Kompella, M. Luciw and J. Schmidhuber. "Incremental 
Slow Feature Analysis: Adaptive Low-Complexity Slow Feature Updating from 
High-Dimensional Input Streams", Neural Computation Journal, 
Vol. 24 (11), pp. 2994--3024, 2012.  

Abstract 
--------

Extract slowly varying components from the input data incrementally.
More information about Incremental Slow Feature Analysis can be found in:

Some of the terminology used in the code is inspired from MDP toolkit
(www.mdp-toolkit.sourceforge.net)

Files
-----

- ccipca.py 	: Candid Covariance-Free Incremental PCA module
- mca.py 		: Minor Component Analysis module
- incsfa.py 	: Incremental Slow Feature Analysis module
- signalstats.py 	: Incremental signal stats modules 
- trainer.py	: trainer module used for training the modules (modes: 'Incremental', 'BlockIncremental', 'Batch')
- test_incsfa.py	: Test example code for IncSFA

Optional Dependencies
---------------------

- MDP toolkit for test_incsfa
- PyQTGraph for fast animated plotting (default is matplotlib)





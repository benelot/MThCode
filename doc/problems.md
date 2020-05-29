## Problems

#### Fundamental questions

* Plausibility of ANN as neural population network.
* Plausibility of iEEG as network output.



#### Technical problems

* How to enhance general performance?
* Performance with non-linearity correspondence constraints.

* Dynamics of full-semi-recurrence.

* Dynamics of output  $\mathbf{u}^{(t)}$ before $\phi$.
  * Probably nothing special.


* Influence of hidden nodes.



#### Tasks

* (1) Find well-grounded choice for window size
  * auto-correlation length $\rightarrow$ window size
  * Fourier spectrum analysis $\rightarrow$ window size
* (2) Make three networks:
  * fully-semi recurrent
  * semi-semi recurrent
  * output recurrent
* (3) Activation function
  * Try $\textrm{relu}$ with positive and negative visible nodes
  * Try $\sigma$ with positive and negative visible nodes
  * Default NN with modified $\sigma$ 


* (4) Evaluate robustness of weights with multiple 40 second segments
* Sleep scoring
* (5) Analyze weights with video of $\mathbf{u}^{(t)}$



#### Ideas

* Introduction of dropout nodes.
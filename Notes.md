Notes about the code and different files in it

# contrastive_training.py
- It first initializes arguments and sets the script to use one GPU
- Runs init experiment, which is responsible for creating the paths for logging and checkpoint saving. It also creates a Tensorboard writer and gives it some of the arguments.
- Then it loads the VIT Dino pre-trained model, and freezes all layers except some of the last blocks (default from 11/11)
- Then it creates the augmentation transformer for the train and test data, and it creates an augmenter to generate multiple views of the training data for contrastive learning
- Then it loads the datasets with these transforms, and creates a sampler to balance labelled and unlabelled classes. The sampler assigns 1 weight to labeled data and len(all_data)/len(unlabeled_data) weight to unlabeled data
- Then the dataloaders are made and the prediction head is created. I have no idea why the prediction head is made alone, and I should **invistigate that further**

## Training
- Pretty much same as in SimGCD, just without mixed precision and using K-Means instead of parametric classification. No clustering training is done in this script
- Unlike in SimGCD it supports running unsupervised contrastive loss only on unlabelled data, while SimGCD can only do it with all data
- SimGCD normalizes projections only for supervised loss, while this one does it for both
- Does accuracy calculation every batch, while SimGCD does it every epoch

## Testing
- Goes through test set and gets predictions for all images. Also creates a labels array and a mask array, which designates each prediction/value to open or closed set
- Fits all data using sklearn's KMeans algorith. Number of clusters is equal to number of classes
- Does not really do any supervised KMeans
- Uses test set as validation set. This probably should not be done and SimGCD removes this code

# estimate_k.py
- Uses a features dataset that takes an original dataset, then replaces the images by the pre-computed features. It loads features on request and not all in the start. Will result in overhead if passing through the dataset more than once.
- Estimates optimal number of classes from the training dataset
- Supports two methods for estimating K:
  - Using an inbuilt scipy optimzer based on Brent's algorithm
  - Using binary search
- Both utilize a test_kmeans function that uses sklearn's inbuil K-Means method. This also does not do any kind of supervised K-Means. Both test_kmeans functions are essentially identical. They just return opposite sign values.
- There are multiple score functions, such as cluster, normalized mutual info (nmi) and adjusted random (ari) score. **I'll read about them later**
- Search algorithms are only concerned with labelled clustering accuracy

# k_means.py
- Creates dataset and target transform just like in contrastive learning, than converts to features dataset
- Has an ominous "TODO: Debug" comment
- Mask_cls is a mask for open set, while mas_lab is a mask for unlabelled data. **I kinda still do not get the difference**

# K_Means Class
- Has a main function for testing, which can be useful.
- Fitting function tries to fit K-Means n_init times, and returns the values from the best K-Means fit
- Each time a K-Means fit is to be run it run the fit_once function with up to max_iterations
- Fit once first assigns the labeled centers as the mean of their corresponding labeled features, then it uses a modified K-Means++ algorithm to initialize the rest of the centers
- During iterations it find the distance between unlabelled features and all centers, while distance of labelled features are only computed with labelled centers
- I will need to **refresh my knowledge on K-Means**, but there seems to only be two distance functions


# Notes
- There is no script that actually tests the computer model and clusters on the test set

# Development
- Understand the contrastive loss and adapt it to hyperbolic space (Look at Meru for reference)
- Currently uses Scikit's K-Means algorithm. Need to adapt the semi-supervised K-Means algorithm in the code to work in hyperbolic space and work with fully supervised data.
- All functions are available in MERU, so I just need to copy them and adapt the model to work with hyperbolic mapping

# Calculations

All formulas going from Lorentz to other models are from the book "Foundations of Hyperbolic Manifolds- John G. Ratcliffe". Poincare is somewhere in chapter 4.5 and Klein is somewhere in chapter 6.1

## Lorentz to Klein mapping
According to https://ieeexplore.ieee.org/abstract/document/9658224 the mapping from Lorentz coordinates to Klein coordinates is:

$$x_K = \frac{x_{space}}{x_{time}} = \frac{x_{space}}{\sqrt{1/c+x_{space}x_{space}^{\intercal}}}$$ 

Not the question is. Can I go back from Klein coordinates to Lorentzian without saving $x_{time}$? This should be possible considering the mapping is one to one.

$$x_K^2=\frac{x_{space}^2}{1/c+x_{space}x_{space}^{\intercal}}$$
$$(1/c+x_{space}x_{space}^{\intercal})x_K^2=x_{space}^2$$
$$x_K^2/c+x_{space}x_{space}^{\intercal}x_K^2=x_{space}^2$$
$$x_K^2/c=x_{space}^2-x_{space}x_{space}^{\intercal}x_K^2$$

Nevermind all that. I found the answer in this paper https://arxiv.org/html/2410.16813v1 

$$x_L=\frac{1}{\sqrt{1+x_{K}x_{K}^{\intercal}}}[1,x_K]$$

Sadly this is only for $c=-1$ ($c=1$ in MERU), meaning I have to find a way to make it work with arbitrary curvature  

We try again:

$$x_K = \frac{x_{space}}{x_{time}} = \frac{x_{space}}{\sqrt{1/c+x_{space}x_{space}^{\intercal}}}$$ 

and

$$x_{time}x_K=x_{space}$$

Then

$$x_K = \frac{x_{time}x_K}{\sqrt{1/c+(x_{time}x_K)(x_{time}x_K)^{\intercal}}} = \frac{x_{time}}{\sqrt{1/c+x_{time}^2(x_Kx_K^{\intercal})}}x_K$$

$$1 = \frac{x_{time}}{\sqrt{1/c+x_{time}^2||x_K||^2}}$$

$$1/c+x_{time}^2||x_K||^2 = x_{time}^2$$

$$1/c = (1-||x_K||^2)x_{time}^2$$

$$x_{time}^2 = \frac{1}{c(1-||x_K||^2)}$$

$$x_{time} = \frac{1}{\sqrt{c-c||x_K||^2}}$$

## Exponential maps through Lorentz

$$x_B = \frac{x_{space}}{1+x_{time}}$$

$$x_{space} = \frac{\sinh(\sqrt{c}||x||)}{\sqrt{c}||x||}x$$

$$x_{time}=\sqrt{1/c+||x_{space}||^2}=\sqrt{1/c+\left|\left|\frac{\sinh(\sqrt{c}||x||)}{\sqrt{c}||x||}x\right|\right|^2}$$

$$x_{time}=\sqrt{1/c+\left(\frac{\sinh(\sqrt{c}||x||)}{\sqrt{c}||x||}\right)^2xx^{\intercal}}=\sqrt{1/c+\frac{\sinh^2(\sqrt{c}||x||)}{c||x||^2}||x||^2}$$

$$x_{time}=\sqrt{1/c+\frac{\sinh^2(\sqrt{c}||x||)}{c}}=\sqrt{\frac{1+\sinh^2(\sqrt{c}||x||)}{c}}=\sqrt{\frac{1}{c}+\frac{(e^{\sqrt{c}||x||}-e^{-\sqrt{c}||x||})^2}{4c}}$$

$$x_B=\frac{\frac{\sinh(\sqrt{c}||x||)}{\sqrt{c}||x||}x}{1+\sqrt{\frac{1+\sinh^2(\sqrt{c}||x||)}{c}}}=\frac{\sinh(\sqrt{c}||x||)}{\sqrt{c}||x||+\frac{\sqrt{1+\sinh^2(\sqrt{c}||x||)}}{c||x||}}x=\frac{\sinh(\sqrt{c}||x||)}{\frac{c\sqrt{c}||x||^2+\sqrt{1+\sinh^2(\sqrt{c}||x||)}}{c||x||}}x$$

$$x_B=\frac{c||x||\sinh(\sqrt{c}||x||)}{c\sqrt{c}||x||^2+\sqrt{1+\sinh^2(\sqrt{c}||x||)}}x$$

I do not have the skills to continue with the Poincare derivation, so I will derive the simpler Klein exponential map instead:

$$x_K=\frac{x_{space}}{x_{time}}=\frac{\frac{\sinh(\sqrt{c}||x||)}{\sqrt{c}||x||}x}{\sqrt{\frac{1+\sinh^2(\sqrt{c}||x||)}{c}}}=\frac{\sinh(\sqrt{c}||x||)}{||x||\sqrt{1+\sinh^2(\sqrt{c}||x||)}}x$$

$$x_K=\frac{\sinh(\sqrt{c}||x||)}{||x||\sqrt{1+\cosh^2(\sqrt{c}||x||)-1}}x=\frac{\sinh(\sqrt{c}||x||)}{||x||\cosh(\sqrt{c}||x||)}x=\frac{tanh(\sqrt{c}||x||)}{||x||}x$$

It worked in the code, and it is very similar to the poincare exponential map in the newer survey

Maybe I can get the Poincare map from the transformation from Klein to Poincare:

$$x_B=\frac{1}{1+\sqrt{1-||x_K||^2}}x_K=\frac{\frac{\tanh(\sqrt{c}||x||)}{||x||}}{1+\sqrt{1-\left(\frac{\tanh(\sqrt{c}||x||)}{||x||}\right)^2||x||^2}}x$$

$$x_B=\frac{\tanh(\sqrt{c}||x||)}{1+\sqrt{1-\tanh^2(\sqrt{c}||x||)}}\frac{x}{||x||}=\frac{\tanh(\sqrt{c}||x||)}{1+\sqrt{cosh^{-2}(\sqrt{c}||x||)}}\frac{x}{||x||}$$

$$x_B=\frac{\tanh(\sqrt{c}||x||)}{1+\cosh^{-1}(\sqrt{c}||x||)}\frac{x}{||x||}=\frac{\frac{\sinh(\sqrt{c}||x||)}{\cosh(\sqrt{c}||x||)}}{\frac{\cosh(\sqrt{c}||x||)+1}{\cosh(\sqrt{c}||x||)}}\frac{x}{||x||}=\frac{\sinh(\sqrt{c}||x||)}{\cosh(\sqrt{c}||x||)+1}\frac{x}{||x||}$$

Got myself some nice material for the report lol

Nevermind. **There is a good chance that the transformations between the models I used are wrong. Because none of them have the curvature in them**. Transformations between Poincare and Klein in equation 23 in new survey have curvature in them.

## Projection with adjusted width
When the curvature changes, so does the Lorentz hyperbola. Higher curvature leads to ~~a wider parabola that has~~ a lower minimum (at $x_{time}1/\sqrt{c}$). ~~This means that if we keep the Klein circle at $x_{time}=1$, then it will not be able to capture the entire hyperbola in a radius of 1~~ Previous statement was not true. The function of the hyperbola's radius is $$r_L=\sqrt{x_{time}^2-1/c}$$, and the function of the radius of the projection view making a circle of radius 1 at $x_{time}=1$ is $r_V=x_{time}$ because:

$$r_L=\sqrt{x_{time}^2-1/c}<\sqrt{x_{time}^2}=x_{time}=r_V$$

Meaning that the hyperbola can always be projected to a circle of radius 1 sitting at $x_{time}=1$

## Final Verdict

Most of the resources I found work with curvature of -1 only. I am not going to investigate further to try to find functions for all curvatures, so I am just going to use the same functions that work with -1 for all. I think they work just fine, and from a projection perspective they should satisfy the following (For Klein at least):
- The transformation from Lorentz to Klein is one to one
- Changing Lorentz curvature should not change the fact that Klein disk has straight line distances

## Scaled Klein Projection

$$x_K = \frac{x_{space}}{c\cdot x_{time}} = \frac{x_{space}}{c\cdot \sqrt{1/c+x_{space}x_{space}^{\intercal}}}$$ 

and

$$c\cdot x_{time}x_K=x_{space}$$

Then

$$x_K = \frac{c\cdot x_{time}x_K}{c\sqrt{1/c+(c\cdot x_{time}x_K)(c\cdot x_{time}x_K)^{\intercal}}} = \frac{x_{time}}{\sqrt{1/c+c^2x_{time}^2(x_Kx_K^{\intercal})}}x_K$$

$$1 = \frac{x_{time}}{\sqrt{1/c+c^2x_{time}^2||x_K||^2}}$$

$$1/c+c^2x_{time}^2||x_K||^2 = x_{time}^2$$

$$1/c = (1-c^2||x_K||^2)x_{time}^2$$

$$x_{time}^2 = \frac{1}{c(1-c^2||x_K||^2)}$$

$$x_{time} = \frac{1}{\sqrt{c-c^3||x_K||^2}}$$

This does not seem right, but then I remember that the Klein ball is now scaled by $1/c$, ($||x_K||\leq 1/c$), so it actually is within bounds. Although technically it is the same as using the normal Klein ball ($||x_K||\leq 1$) for the calculation.

Essentially this is like going back to the unit Klein ball before doing time retrieval

## Scaled Klein Projection 2

$$x_K = \frac{x_{space}}{\sqrt{c}\cdot x_{time}} = \frac{x_{space}}{\sqrt{c}\cdot \sqrt{1/c+x_{space}x_{space}^{\intercal}}}$$ 

and

$$\sqrt{c}\cdot x_{time}x_K=x_{space}$$

Then

$$x_K = \frac{\sqrt{c}\cdot x_{time}x_K}{\sqrt{c}\cdot \sqrt{1/c+(\sqrt{c}\cdot x_{time}x_K)(\sqrt{c}\cdot x_{time}x_K)^{\intercal}}} = \frac{x_{time}}{\sqrt{1/c+cx_{time}^2(x_Kx_K^{\intercal})}}x_K$$

$$1 = \frac{x_{time}}{\sqrt{1/c+cx_{time}^2||x_K||^2}}$$

$$1/c+cx_{time}^2||x_K||^2 = x_{time}^2$$

$$1/c = (1-c||x_K||^2)x_{time}^2$$

$$x_{time}^2 = \frac{1}{c(1-c||x_K||^2)}$$

$$x_{time} = \frac{1}{\sqrt{c-c^2||x_K||^2}}$$

This time maximum value of $||x_K||$ is $1/\sqrt{c}$, meaning that the denominator will approach 0 as $||x_K||$ approaches $1/\sqrt{c}$ and $x_{time}$ will approach $\infty$

## Einstein midpoint straight from Lorentz

Note that the Lorentz factor is:

$$\gamma=\frac{1}{\sqrt{1-c||x_K||^2}}=\sqrt{c}x_{time}$$

Because from previous derivation we have:

$$x_{time} = \frac{1}{\sqrt{c-c^2||x_K||^2}}$$

The Einstein midpoint formula is:

$$P=\frac{\sum_{i=1}^{B}{\gamma_i x_{K,i}}}{\sum_{i=1}^{B}{\gamma_i}}=\frac{\sum_{i=1}^{B}{\sqrt{c}x_{time, i}\frac{x_{space, i}}{\sqrt{c}x_{time,i}}}}{\sum_{i=1}^{B}{\sqrt{c}x_{time,i}}}=\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sum_{i=1}^{B}{\sqrt{c}x_{time,i}}}=\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sqrt{c}\sum_{i=1}^{B}{x_{time,i}}}$$

$$P=\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sum_{i=1}^{B}{\sqrt{c}x_{time,i}}}=\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sum_{i=1}^{B}{\sqrt{c}\sqrt{1/c+||x_{space,i}||^2}}}=\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sum_{i=1}^{B}{\sqrt{1+c||x_{space}||^2}}}$$

Mohamad from 2 months later: I just realized that this Einstein midpoint is still in Klein space and I need project it back to Lorentz. Oops, so here I go:

$$x_{time}=\frac{1}{\sqrt{c-c^2||P||^2}}=\left(\sqrt{c-c^2\left||\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sum_{i=1}^{B}{\sqrt{c}x_{time,i}}}|\right|^2}\right)^{-1}$$

$$x_{time}=\left(\sqrt{c-\left(\frac{c}{\sum_{i=1}^{B}{\sqrt{c}x_{time,i}}}\right)^2\left||\sum_{i=1}^{B}{x_{space, i}}|\right|^2}\right)^{-1}$$

$$x_{time}=\left(\sqrt{c-\frac{c}{(\sum_{i=1}^{B}{x_{time,i}})^2}\left||\sum_{i=1}^{B}{x_{space, i}}|\right|^2}\right)^{-1}$$

$$x_{time}=\left(\sqrt{c}\sqrt{1-\frac{1}{(\sum_{i=1}^{B}{x_{time,i}})^2}\left||\sum_{i=1}^{B}{x_{space, i}}|\right|^2}\right)^{-1}$$

With this we can find $x_{space}$:
$$x_{space}=\sqrt{c}x_{time}P=\frac{\sqrt{c}}{\sqrt{c-\frac{c}{(\sum_{i=1}^{B}{x_{time,i}})^2}\left||\sum_{i=1}^{B}{x_{space, i}}|\right|^2}}\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sqrt{c}\sum_{i=1}^{B}{x_{time,i}}}$$

$$x_{space}=\frac{\sum_{i=1}^{B}{x_{space, i}}}{\sum_{i=1}^{B}{x_{time,i}}\sqrt{c-\frac{c}{(\sum_{i=1}^{B}{x_{time,i}})^2}\left||\sum_{i=1}^{B}{x_{space, i}}|\right|^2}}$$
## Klein exponential map from Poincare

**Note: All the following only works with the map at origin**

Poincare exponential map is the following (new survey):

$$x_B=\tanh(\sqrt{c||x||^2})\frac{x}{\sqrt{c||x||^2}}$$

This is for the Poincare model where $\{||x_B|| < 1/\sqrt{c}\}$. The transformation from Poincare to Klein is the following (new survey):

$$x_K=\frac{2}{1+c||x_B||^2}x_B$$

This should lead to the Klein model where $\{||x_K|| < 1/\sqrt{c}\}$. Inserting the exponential map here we get:

$$x_K=\frac{2}{1+c\left||\tanh(\sqrt{c||x||^2})\frac{x}{\sqrt{c||x||^2}}|\right|^2}\tanh(\sqrt{c||x||^2})\frac{x}{\sqrt{c||x||^2}}$$

$$x_K=\frac{2}{1+c\left(\tanh(\sqrt{c||x||^2})\frac{1}{\sqrt{c||x||^2}}\right)^2||x||^2}\tanh(\sqrt{c||x||^2})\frac{x}{\sqrt{c||x||^2}}$$

$$x_K=\frac{2}{1+c\tanh^2(\sqrt{c||x||^2})\frac{1}{c}}\tanh(\sqrt{c||x||^2})\frac{x}{\sqrt{c||x||^2}}$$

$$x_K=\frac{2\tanh(\sqrt{c||x||^2})}{1+\tanh^2(\sqrt{c||x||^2})}\frac{x}{\sqrt{c||x||^2}}=\tanh(2\sqrt{c||x||^2})\frac{x}{\sqrt{c||x||^2}}$$

The Poincare to Klein projection is a function that turns $\frac{\tanh(\sqrt{c||x||^2})}{\sqrt{c||x||^2}}x$ to $\frac{\tanh(2\sqrt{c||x||^2})}{\sqrt{c||x||^2}}x$

The projection from Lorentz to Klein turns the Lorentz exponential map to $\frac{\tanh(\sqrt{c||x||^2})}{\sqrt{||x||^2}}x$ meaning that to turn it into $\frac{\tanh(2\sqrt{c||x||^2})}{\sqrt{c||x||^2}}x$ we have to divide by $\sqrt{c}$ and then pass it through the Poincare to Klein transformation. Turning the Lorentz to Klein transformation into:
$$x_K=\frac{2}{1+c\left||\frac{x_{space}}{\sqrt{c}x_{time}}|\right|^2}\frac{x_{space}}{\sqrt{c}x_{time}}$$

Irrelevant stuff:

$$\tanh(xy)=\frac{e^{xy}-e^{-xy}}{e^{xy}+e^{-xy}}=\frac{y\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}}{1+\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}^y}$$

$$\tanh(xy)=\tanh(x+x(y-1))=\frac{\tanh(x)+\tanh(x(y-1))}{1+\tanh(x)\tanh(x(y-1))}$$

$$tanh(xy)=\frac{\tanh(x)+\tanh(xy-x)}{1+\tanh(x)\tanh(xy-x)}=\frac{\tanh(x)+\tanh(x+x(y-2))}{1+\tanh(x)\tanh(x+x(y-2))}$$

This continues on recursively. Oops

## Getting x_time from new transform

$$x_{time}=\sqrt{1/c+||x_{space}||^2}\leftrightarrow ||x_{space}||^2=x_{time}^2-1/c$$

and

$$x_K = \frac{2}{1+c\left||\frac{x_{space}}{\sqrt{c}x_{time}}|\right|^2}\frac{x_{space}}{\sqrt{c}x_{time}}$$

$$x_K = \frac{2}{1+\frac{c}{cx_{time}^2}||x_{space}||^2}\frac{x_{space}}{\sqrt{c}x_{time}}=\frac{2x_{space}}{\sqrt{c}x_{time}+\frac{\sqrt{c}}{x_{time}}||x_{space}||^2}$$

$$x_K=\frac{2x_{space}}{\sqrt{c}\sqrt{1/c+||x_{space}||^2}+\frac{\sqrt{c}}{\sqrt{1/c+||x_{space}||^2}}||x_{space}||^2}$$

and

$$\frac{\sqrt{c}x_{time}+\frac{\sqrt{c}}{x_{time}}||x_{space}||^2}{2}x_K=\frac{\sqrt{c}x_{time}+\frac{\sqrt{c}}{x_{time}}(x_{time}^2-\frac{1}{c})}{2}x_K=x_{space}$$

$$x_{space}=\frac{\sqrt{c}x_{time}+\sqrt{c}x_{time}-\frac{1}{\sqrt{c}x_{time}}}{2}x_K=\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K$$

Then

$$x_K=\frac{2}{\sqrt{1+c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2}+\frac{c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2}{\sqrt{1+c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2}}}\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K$$

$$1=\frac{2}{\frac{1+c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2+c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2}{\sqrt{1+c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2}}}\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}$$

$$1=\frac{2\sqrt{1+c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2}}{1+2c\left||\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}x_K|\right|^2}\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}$$

$$1=\frac{2\sqrt{1+c\left(\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}\right)^2||x_K||^2}}{1+2c\left(\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}\right)^2||x_K||^2}\frac{2cx_{time}^2-1}{2\sqrt{c}x_{time}}$$

Maaaaaaaaaaaaaaaan this is so complicated

# Fixing NaN in loss

So far the the main culprit is getting an outlier hyperbolic embedding with a huge norm, leading to the calculating distances to overflow. However, this is not always the case, that is why I am doing more testing.

From the distance function:
$$d_L(x,y)=\sqrt{\frac{1}{c}}\cosh^{-1}(-c<x,y>_L)$$

We can see that lower curvature leads to higher distances. That is because for high inner products, the c inside the arccosh does not reduce its value by much. Meaning that models with lower curvature have a lower threshold for the norm of the hyperbolic embeddings.

Of course I am using negative distance, so the value being passed to the exponential function right after are all under 0. I have tried investigating the supconloss function and can see that at certain distances the values coming out of the exponent are extremely small, and are reaching 0 at the end.

The full loss function is as follows:

$$-\frac{1}{|N(i)|}\sum_{j\in N(i)}{\log\left(\frac{e^{-d_L(x_i, x_j)}}{\sum^N_{k=0}{1_{k!=i}e^{-d_L(x_i, x_k)}}}\right)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}{-\log\left(\frac{e^{-d_L(x_i, x_j)}}{\sum^N_{k=0}{1_{k!=i}e^{-d_L(x_i, x_k)}}}\right)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}{\log\left(\frac{\sum^N_{k=0}{1_{k!=i}e^{-d_L(x_i, x_k)}}}{e^{-d_L(x_i, x_j)}}\right)}$$

With high distance values the exponents in the loss are gonna be extremely small, and if one embedding is very far, then its row will be all 0 and the denominator will become 0 leading to problems.

After investigating it seems that a norm of 19 is the maximum I can go before having a risk of NaN. This was done by taking two opposite vectors and increasing their radius, where it produces NaN at 20. However, that was with 1.0 curvature. If I use 0.1 curvature then it reduces to 4. Now the question is: Do I want to have a hard clamp on 4? Or do I want to have a loss function that might lead to training a proper proj_alpha? Or do I want to do mixed precision calculations for calulating the loss? Maybe add a loss depending on the maximum distance? Maybe add an epsilon number so that my denominator does not go under the minimum floating number?

Another thought from this: The model is probably reducing the curvature because it is a great way to increase the overall distance between embeddings.

It is a bit counter-intuitive for me that a lower curvature results in embeddings being mapped farther apart in hyperbolic space, but that's just how it is the distance function works, lower c leads to steeper growth in the function.

I have tried adding an epsilon making the loss function to:
$$-\frac{1}{|N(i)|}\sum_{j\in N(i)}{\log\left(\frac{e^{-d_L(x_i, x_j)}}{\left(\sum^N_{k=0}{1_{k!=i}e^{-d_L(x_i, x_k)}}\right)+\varepsilon}\right)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}{\log\left(\frac{\left(\sum^N_{k=0}{1_{k!=i}e^{-d_L(x_i, x_k)}}\right)+\varepsilon}{e^{-d_L(x_i, x_j)}}\right)}$$

It is very easy for values to 

## Poincare distance simplification

The Poincare distance is given as:

$$D_P(x,y,c)=\frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}||-x\oplus_c y||)$$

Where:

$$x\oplus_c y=\frac{(1-2c<x,y>-c||y||^2)x+(1+c||x||^2)y}{1-2c<x,y>+c^2||x||^2||y||^2}, c<0$$

In the code I can see that the distance is simplified to:

$$D_P(x,y,c)=\frac{2}{\sqrt{c}}\tanh^{-1}{\left(\frac{||x||^2-2<x,y>+||y||^2}{1+2c<x,y>+c^2||x||^2||y||^2}\right)}$$

I am assuming this simplification requires c>0. Meaning that this simplification occured:

$$\sqrt{c}\left||\frac{-(1+2c<-x,y>+c||y||^2)x+(1-c||x||^2)y}{1+2c<-x,y>+c^2||x||^2||y||^2}|\right|=\frac{||x||^2-2<x,y>+||y||^2}{1+2c<x,y>+c^2||x||^2||y||^2}$$

I will try to see how this is done here with c>0:

$$\sqrt{c}\left||\frac{-(1+2c<-x,y>+c||y||^2)x+(1-c||x||^2)y}{1+2c<-x,y>+c^2||x||^2||y||^2}|\right|=$$
$$=\frac{\sqrt{c}}{1+2c<-x,y>+c^2||x||^2||y||^2}||-(1+2c<-x,y>+c||y||^2)x+(1-c||x||^2)y||=$$
$$=K||-(1+2c<-x,y>+c||y||^2)x+(1-c||x||^2)y||$$

I cannot really see how they did it, and I am not trusting a formula I cannot reach myself. Furthermore, both the hyperbolic vision survey and the hyperbolic geometry books have a different distance formula that comes from transforming from Lorentz space, while this one comes from Mobius algebra. Both can be viable, but I will work with the one from the book:

$$D_P(x,y,c)=\frac{1}{\sqrt{c}}\cosh^{-1}{\left(1+\frac{2||x-y||^2}{(1-||x||^2)(1-||y||^2)}\right)}$$

I will need to figure out a way to do the numerator in pytorch, but the rest should be simple.

## Einstein midpoint in Poincare

The transformations between Poincare and Klein are:

$$x_K = \frac{2x_P}{1+c||x_P||^2}$$

$$x_P = \frac{x_K}{1+\sqrt{1-c||x_K||^2}}$$

The Einstein midpoint in Klein is:

$$P = \frac{\sum^B_{i=1}{\gamma_i x_{K,i}}}{\sum^B_{i=1}{\gamma_i}} = \frac{\sum^B_{i=1}{\gamma_i \frac{2x_{P,i}}{1+c||x_{P,i}||^2}}}{\sum^B_{i=1}{\gamma_i}}$$

and:

$$\gamma_i = \frac{1}{\sqrt{1-c||x_{K,i}||^2}}=\left(\sqrt{1-c\left||\frac{2x_{P,i}}{1+c||x_{P,i}||^2}|\right|^2}\right)^{-1}$$

$$\gamma_i = \left(\sqrt{1-\frac{4c}{(1+c||x_{P,i}||^2)^2}||x_{P,i}||^2}\right)^{-1}$$

To be honest. I think the best way to go about this is to convert all to Klein, find the midpoint and convert back. There is not much simplification here.

## Understanding the Lorentz Centroid
I have two sources for the Lorentz Centroid. The first is the paper Lorentzian distance learning for hyperbolic representations, and the second is the following math exchange answer https://math.stackexchange.com/a/2173370. The Lorentz centroid is supposidely the point that minimizes the squared Lorentz distance to a batch of points, and is found using:

$$\mu = \sqrt{\Beta}\frac{\sum_{i=1}^{B}{v_ix_{L,i}}}{|||\sum_{i=1}^{B}{v_ix_{L,i}}||_{\mathcal{L}}|}$$

Where $x_{L,i}$ are point in the hyperboloid ($x_{space} | x_{time}$), $v_i$ are weights and $\Beta = 1/c$. The operation $|||a||_{\mathcal{L}}|$ is the modulus of the Lorentz norm, but the Lorentz norm of a vector is:
$$||a||_{\mathcal{L}}=\sqrt{<a,a>_{\mathcal{L}}}=\sqrt{-1/c}$$

Nvm that is not correct because the vector $a$ doesn't have to be on the hyperboloid, it just has to be time-like (Within light cone and has negative Lorentz inner product). So the formula seems to be alright, but can we simplifiy it? My initial thought is that we cannot really because we need to find $x_{time}$ for $\sum_{i=1}^{B}{v_ix_{L,i}}$ to get it's modulus norm. But how do we get that modulus norm? It involves squaring a negative number. What kind of imaginary number do we get from squaring a negative number? Well I think it is kinda simple here is what I think it is:

$$\sqrt{-x}=\sqrt{i^2x}=i\sqrt{x}$$

But how do we get the modulus of this value? It has no real component? Well seems I am remembering wrong. The modulus of an imaginary number is:

$$|x+iy|=\sqrt{x^2+y^2}$$

So I just get $\sqrt{\sqrt{x}^2}=\sqrt{x}$ as my final answer. I can simplify my calculations to this:

$$\mu_{space}=\frac{1}{\sqrt{c}}\frac{\sum_{i=1}^{B}{v_ix_{space,i}}}{\sqrt{|||\sum_{i=1}^{B}{v_ix_{space,i}}||^2-(\sum_{i=1}^{B}{v_ix_{time,i}})^2|}}$$

But that doesn't really matter since I gotta find the full vector anyways. I will also use a weight $v_i=1/B$ to introduce numerical stability to it. I should demonstrate the difference between all the centroids in my project. Just make a circle in Lorentz space and show the computed centroids
# Branch Explanation
The branch try to solve the GPU OOM issue.

Below are the modification to avoid OOM:
- Implement the training docker image for tf2.2.x training & profile
- Remove create Word Formulation in model
- Create a small [English+Num charsets](config/charset.alphabeta.txt) to test the code logic
- Implement the loss & metric by dict style(but only supported by TF2.2.x, TF2.1.x not support)
- Try to debug batch size for OOM, finally got proper size: 7.
- Try to profile by tensorboard, found only tf2.2.x support

# Some Details

1. I don't wang to update the server GPU driver, CUDA and cuDNN version. 
So, I made my training docker image inherit from tensorflow/tensorflow:2.1.0-gpu-py3.
But I do need tf2.2.x, but there is no available image for 2.2.x, 
So, I build one basing on tf2.2.1 image.

2.By tf2.2.x, I can profile the training process, but unfortunately, 
I still cannot find the GPU memeory usage context information.

3.I found the model can return dict style result, 
which can later be used for loss and metrics, and can make the code more readable, 
but it need the tf2.2.x, that is the reason I made the tf2.2.x training docker image.  
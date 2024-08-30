Training on a GPU
=================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 15 min

        **Objectives:**
            #. Learn how to traing the model on a GPU.
            #. Learn how to save a model.
            #. Learn how to load a saved model. 

Set the default device
**********************

We can set a default device when building a model, ensuring that all operations occur on this device. If available, we can set the GPU as the default device.

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Saving and Loading a Model
******************************

We can save the model in a specific path in the syste.

.. code-block:: python

    modelpath = os.path.expandvars('/home/$USER/class_model')
    torch.save(class_model.state_dict(), modelpath)

This saved model can be loaded when needed. During loading, you can directly specify the device using the `map_location` parameter or move the model to the 
desired device afterward using the `.to()` function.

.. code-block:: python

    class_model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    class_model.to(device)

Training the Model
******************

When training, both the model and all the data it operates on should be on the same device.

.. code-block:: python

    n_epochs = 100
    batch_size = 10
 
    for epoch in range(n_epochs):
        for i in range(0, len(X_tensor), batch_size):
            Xbatch = X_tensor[i:i+batch_size].to(device) # move the tensor to GPU

            y_pred = class_model(Xbatch)
        
            ybatch = y_tensor[i:i+batch_size].to(device) # move the tensor to GPU
        
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

.. admonition:: Exercise
   :class: todo

    Try the notebook *GPU_NN.ipynb*.


.. admonition:: Key Points
   :class: hint

    #. We can set a default device in PyTorch.
    #. During training, ensure that both the model and the data it operates on are on the same device.



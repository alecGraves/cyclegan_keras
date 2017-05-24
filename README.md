# cyclegan_keras
cyclegan reimplimented in keras

## What has been made?
* Instance Norm  layer for keras
* Cycle loss function for keras
* Test cyclegan training script

## What needs to be done?
* Bigger models for non-mnist images need to be created

## Pictures!
For the test training script, I decided to have cyclegan learn a mapping from cat pictures to mnist pictures (and back).
This took like 10 hours on a Nvidia GTX 1060 using the training parameters in mnist_test.py.

Here are some results (from various stages of training):

![19](/test/example_output/images/19.png)

![31](/test/example_output/images/31.png)

![34](/test/example_output/images/34.png)

![42](/test/example_output/images/42.png)

![62](/test/example_output/images/62.png)

![72](/test/example_output/images/72.png)

![90](/test/example_output/images/90.png)

